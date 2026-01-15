//! Parquet storage engine implementation.

use anyhow::Result;
use arrow::array::RecordBatch;
use arrow::datatypes::SchemaRef;
use async_trait::async_trait;
use indicatif::{ProgressBar, ProgressStyle};
use parquet::arrow::arrow_reader::{
    ArrowReaderMetadata, ArrowReaderOptions, ParquetRecordBatchReaderBuilder, RowSelection,
    RowSelector,
};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::{EnabledStatistics, WriterProperties};
use parquet::file::reader::{ChunkReader, FileReader, Length, SerializedFileReader};
use std::fs::{self, File};
use std::io::BufReader;
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::sync::Arc;
use tokio::runtime::Runtime;

use crate::cache::drop_directory_cache;
use crate::data::{create_schema, generate_vector_batch};
use crate::Config;

use super::traits::{DatasetHandle, Engine};

struct FileRef {
    file: Arc<File>,
    size: u64,
}

impl Length for FileRef {
    fn len(&self) -> u64 {
        self.size
    }
}

impl ChunkReader for FileRef {
    // This doesn't matter, we never use it
    type T = BufReader<File>;

    fn get_read(&self, _: u64) -> parquet::errors::Result<Self::T> {
        panic!("Not implemented");
    }

    fn get_bytes(&self, start: u64, length: usize) -> parquet::errors::Result<bytes::Bytes> {
        let mut buf = vec![0; length];
        self.file
            .read_exact_at(&mut buf, start)
            .map(|_| bytes::Bytes::from(buf))
            .map_err(|e| parquet::errors::ParquetError::External(e.into()))
    }
}

/// Handle to an open Parquet dataset with cached file handle and metadata.
pub struct ParquetHandle {
    /// Cached file handle (we clone it for each read)
    file: Arc<File>,
    /// Size of the file, in bytes
    size: u64,
    /// Cached Arrow reader metadata
    arrow_metadata: ArrowReaderMetadata,
    /// Cached schema
    schema: SchemaRef,
    /// Total row count
    row_count: usize,
}

impl ParquetHandle {
    fn new(path: &str) -> Result<Self> {
        let file = Arc::new(File::open(path)?);

        let size = file.metadata()?.len();

        let options = ArrowReaderOptions::new().with_page_index(true);

        // Load and cache Arrow reader metadata
        let arrow_metadata = ArrowReaderMetadata::load(file.as_ref(), options)?;
        let schema = arrow_metadata.schema().clone();

        // Get total row count from metadata
        let row_count: usize = arrow_metadata
            .metadata()
            .row_groups()
            .iter()
            .map(|rg| rg.num_rows() as usize)
            .sum();

        Ok(Self {
            file,
            size,
            arrow_metadata,
            schema,
            row_count,
        })
    }
}

/// Convert sorted indices to a RowSelection.
/// Indices must be sorted in ascending order.
fn indices_to_row_selection(indices: &[u64], total_rows: usize) -> RowSelection {
    if indices.is_empty() {
        return RowSelection::from(vec![RowSelector::skip(total_rows)]);
    }

    let mut selectors = Vec::with_capacity(indices.len() * 2);
    let mut current_pos: usize = 0;

    for &idx in indices {
        let idx = idx as usize;

        // Skip rows before this index
        if idx > current_pos {
            selectors.push(RowSelector::skip(idx - current_pos));
        }

        // Select this row
        selectors.push(RowSelector::select(1));
        current_pos = idx + 1;
    }

    // Skip any remaining rows
    if current_pos < total_rows {
        selectors.push(RowSelector::skip(total_rows - current_pos));
    }

    RowSelection::from(selectors)
}

#[async_trait]
impl DatasetHandle for ParquetHandle {
    async fn take(&self, indices: &[u64]) -> Result<RecordBatch> {
        // Build row selection from indices
        let selection = indices_to_row_selection(indices, self.row_count);

        let file = FileRef {
            file: self.file.clone(),
            size: self.size,
        };

        // Build reader with cloned file handle and cached metadata, applying row selection
        let builder =
            ParquetRecordBatchReaderBuilder::new_with_metadata(file, self.arrow_metadata.clone())
                .with_row_selection(selection);
        let reader = builder.build()?;

        // Read selected batches
        let batches: Vec<RecordBatch> = reader.collect::<Result<Vec<_>, _>>()?;

        if batches.is_empty() {
            anyhow::bail!("No data in parquet file");
        }

        // Concatenate batches (should already have only selected rows)
        let result = arrow::compute::concat_batches(&self.schema, &batches)?;
        Ok(result)
    }
}

/// Parquet storage engine.
pub struct ParquetEngine {
    runtime: Arc<Runtime>,
}

impl ParquetEngine {
    pub fn new() -> Self {
        Self {
            runtime: Arc::new(
                tokio::runtime::Builder::new_current_thread()
                    .build()
                    .unwrap(),
            ),
        }
    }

    /// Extract the file path from a URI.
    fn uri_to_path<'a>(&self, uri: &'a str) -> &'a str {
        if let Some(path) = uri.strip_prefix("file://") {
            path
        } else if uri.contains("://") {
            // For other schemes, return as-is (won't work but provides error context)
            uri
        } else {
            // Plain path
            uri
        }
    }

    /// Get the parquet file path within the dataset directory.
    fn get_parquet_file(&self, uri: &str) -> String {
        let base_path = self.uri_to_path(uri);
        format!("{}/data.parquet", base_path)
    }
}

impl Default for ParquetEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Engine for ParquetEngine {
    fn name(&self) -> &'static str {
        "parquet"
    }

    fn runtime(&self) -> Arc<Runtime> {
        self.runtime.clone()
    }

    fn exists(&self, uri: &str, expected_rows: usize) -> bool {
        let parquet_file = self.get_parquet_file(uri);
        let path = Path::new(&parquet_file);

        if !path.exists() {
            return false;
        }

        // Check row count
        if let Ok(file) = File::open(path) {
            if let Ok(reader) = SerializedFileReader::new(file) {
                let metadata = reader.metadata();
                let row_count: usize = metadata
                    .row_groups()
                    .iter()
                    .map(|rg| rg.num_rows() as usize)
                    .sum();
                return row_count == expected_rows;
            }
        }
        false
    }

    fn open(&self, uri: &str) -> Result<Arc<dyn DatasetHandle>> {
        let parquet_file = self.get_parquet_file(uri);
        let handle = ParquetHandle::new(&parquet_file)?;
        Ok(Arc::new(handle))
    }

    fn write(&self, uri: &str, config: &Config) -> Result<Arc<dyn DatasetHandle>> {
        let base_path = self.uri_to_path(uri);
        let parquet_file = self.get_parquet_file(uri);

        println!("\nGenerating dataset: {}", parquet_file);

        // Create the directory
        fs::create_dir_all(base_path)?;

        let num_batches = config.rows_per_dataset / config.write_batch_size;
        let pb = ProgressBar::new(num_batches as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("  Writing batches [{bar:40}] {pos}/{len}")
                .unwrap(),
        );

        let schema = create_schema(config.vector_dim);

        // Create the parquet writer
        let file = File::create(&parquet_file)?;
        let props = WriterProperties::builder()
            .set_dictionary_enabled(false)
            .set_data_page_size_limit(8 * 1024)
            .set_statistics_enabled(EnabledStatistics::None)
            .set_write_batch_size(1)
            .build();
        let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props))?;

        // Write batches
        for _ in 0..num_batches {
            let batch =
                generate_vector_batch(schema.clone(), config.write_batch_size, config.vector_dim)?;
            writer.write(&batch)?;
            pb.inc(1);
        }

        writer.close()?;
        pb.finish();

        // Open the written file with cached handle and metadata
        let handle = ParquetHandle::new(&parquet_file)?;
        Ok(Arc::new(handle))
    }

    fn drop_cache(&self, uri: &str) -> Result<()> {
        let path = self.uri_to_path(uri);
        drop_directory_cache(Path::new(path))
    }
}
