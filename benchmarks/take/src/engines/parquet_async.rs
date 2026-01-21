//! Async Parquet storage engine implementation using tokio I/O.

use anyhow::Result;
use arrow::array::RecordBatch;
use arrow::datatypes::SchemaRef;
use async_trait::async_trait;
use futures::TryStreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use parquet::arrow::arrow_reader::{
    ArrowReaderMetadata, ArrowReaderOptions, RowSelection, RowSelector,
};
use parquet::arrow::async_reader::ParquetRecordBatchStreamBuilder;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::{EnabledStatistics, WriterProperties};
use parquet::file::reader::{FileReader, SerializedFileReader};
use std::fs::{self, File};
use std::path::Path;
use std::sync::Arc;
use tokio::fs::File as TokioFile;
use tokio::runtime::Runtime;

use crate::cache::drop_directory_cache;
use crate::data::{create_schema, generate_vector_batch};
use crate::Config;

use super::traits::{DatasetHandle, Engine};

/// Handle to an open Parquet dataset for async reading.
/// Stores the path and metadata, opens a new file handle per read.
pub struct ParquetAsyncHandle {
    /// Path to the parquet file (for reopening)
    path: String,
    /// Cached Arrow reader metadata
    arrow_metadata: ArrowReaderMetadata,
    /// Cached schema
    schema: SchemaRef,
    /// Total row count
    row_count: usize,
}

impl ParquetAsyncHandle {
    async fn new(path: &str) -> Result<Self> {
        let mut file = TokioFile::open(path).await?;
        let options = ArrowReaderOptions::new().with_page_index(true);

        // Load and cache Arrow reader metadata
        let arrow_metadata = ArrowReaderMetadata::load_async(&mut file, options).await?;
        let schema = arrow_metadata.schema().clone();

        // Get total row count from metadata
        let row_count: usize = arrow_metadata
            .metadata()
            .row_groups()
            .iter()
            .map(|rg| rg.num_rows() as usize)
            .sum();

        Ok(Self {
            path: path.to_string(),
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
impl DatasetHandle for ParquetAsyncHandle {
    async fn take(&self, indices: &[u64]) -> Result<RecordBatch> {
        // Build row selection from indices
        let selection = indices_to_row_selection(indices, self.row_count);

        // Open a new file handle for this read
        let file = TokioFile::open(&self.path).await?;

        // Build async reader with cached metadata, applying row selection
        let builder = ParquetRecordBatchStreamBuilder::new_with_metadata(
            file,
            self.arrow_metadata.clone(),
        )
        .with_row_selection(selection);
        let stream = builder.build()?;

        // Read selected batches asynchronously
        let batches: Vec<RecordBatch> = stream.try_collect().await?;

        if batches.is_empty() {
            anyhow::bail!("No data in parquet file");
        }

        // Concatenate batches
        let result = arrow::compute::concat_batches(&self.schema, &batches)?;
        Ok(result)
    }
}

/// Async Parquet storage engine using tokio I/O.
pub struct ParquetAsyncEngine {
    runtime: Arc<Runtime>,
}

impl ParquetAsyncEngine {
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

impl Default for ParquetAsyncEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Engine for ParquetAsyncEngine {
    fn name(&self) -> &'static str {
        "parquet-async"
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
        // Use block_on to create the async handle
        let handle = self.runtime.block_on(ParquetAsyncHandle::new(&parquet_file))?;
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

        // Create the parquet writer (sync write is fine for benchmarks)
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

        // Open the written file with async handle
        let handle = self.runtime.block_on(ParquetAsyncHandle::new(&parquet_file))?;
        Ok(Arc::new(handle))
    }

    fn drop_cache(&self, uri: &str) -> Result<()> {
        let path = self.uri_to_path(uri);
        drop_directory_cache(Path::new(path))
    }
}
