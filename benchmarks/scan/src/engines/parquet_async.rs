//! Async Parquet storage engine implementation for scan benchmark.

use anyhow::Result;
use arrow::record_batch::RecordBatch;
use async_trait::async_trait;
use futures::TryStreamExt;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::async_reader::ParquetRecordBatchStreamBuilder;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use std::fs::{self, File};
use std::path::Path;
use std::sync::Arc;
use tokio::fs::File as TokioFile;
use tokio::runtime::Runtime;

use crate::cache::drop_directory_cache;

use super::traits::{ScanEngine, ScanHandle};

/// Handle to an open Parquet file for async scanning.
pub struct ParquetAsyncScanHandle {
    /// Path to the parquet file
    path: String,
    /// Total row count (cached)
    row_count: usize,
    /// File size in bytes
    byte_size: u64,
}

impl ParquetAsyncScanHandle {
    fn new(path: &str) -> Result<Self> {
        let file = File::open(path)?;
        let metadata = file.metadata()?;
        let byte_size = metadata.len();

        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let row_count = builder.metadata().file_metadata().num_rows() as usize;

        Ok(Self {
            path: path.to_string(),
            row_count,
            byte_size,
        })
    }
}

#[async_trait]
impl ScanHandle for ParquetAsyncScanHandle {
    async fn scan(&self) -> Result<Vec<RecordBatch>> {
        let file = TokioFile::open(&self.path).await?;
        let builder = ParquetRecordBatchStreamBuilder::new(file).await?;
        let stream = builder.build()?;

        let batches: Vec<RecordBatch> = stream.try_collect().await?;
        Ok(batches)
    }

    fn row_count(&self) -> usize {
        self.row_count
    }

    fn byte_size(&self) -> u64 {
        self.byte_size
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
            uri
        } else {
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

impl ScanEngine for ParquetAsyncEngine {
    fn name(&self) -> &'static str {
        "parquet-async"
    }

    fn runtime(&self) -> Arc<Runtime> {
        self.runtime.clone()
    }

    fn exists(&self, uri: &str) -> bool {
        let parquet_file = self.get_parquet_file(uri);
        Path::new(&parquet_file).exists()
    }

    fn open(&self, uri: &str) -> Result<Arc<dyn ScanHandle>> {
        let parquet_file = self.get_parquet_file(uri);
        let handle = ParquetAsyncScanHandle::new(&parquet_file)?;
        Ok(Arc::new(handle))
    }

    fn write(&self, uri: &str, batches: &[RecordBatch]) -> Result<Arc<dyn ScanHandle>> {
        let base_path = self.uri_to_path(uri);
        let parquet_file = self.get_parquet_file(uri);

        // Create the directory
        fs::create_dir_all(base_path)?;

        // Get schema from first batch
        let schema = batches
            .first()
            .ok_or_else(|| anyhow::anyhow!("No batches to write"))?
            .schema();

        // Create the parquet writer (sync write is fine for setup)
        let file = File::create(&parquet_file)?;
        let props = WriterProperties::builder()
            .set_compression(parquet::basic::Compression::ZSTD(Default::default()))
            .build();
        let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;

        // Write all batches
        for batch in batches {
            writer.write(batch)?;
        }

        writer.close()?;

        // Open the written file
        let handle = ParquetAsyncScanHandle::new(&parquet_file)?;
        Ok(Arc::new(handle))
    }

    fn drop_cache(&self, uri: &str) -> Result<()> {
        let path = self.uri_to_path(uri);
        drop_directory_cache(Path::new(path))
    }
}
