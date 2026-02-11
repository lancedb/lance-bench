//! Lance storage engine implementation for scan benchmark.

use anyhow::Result;
use arrow::array::RecordBatchIterator;
use arrow::record_batch::RecordBatch;
use async_trait::async_trait;
use futures::TryStreamExt;
use lance::dataset::{Dataset, WriteMode, WriteParams};
use lance_file::version::LanceFileVersion;
use std::path::Path;
use std::sync::Arc;
use tokio::runtime::Runtime;
use walkdir::WalkDir;

use crate::cache::drop_directory_cache;

use super::traits::{ScanEngine, ScanHandle};

/// Handle to an open Lance dataset for scanning.
pub struct LanceScanHandle {
    /// The opened Lance dataset
    dataset: Dataset,
    /// Total row count (cached)
    row_count: usize,
    /// Dataset size in bytes (cached)
    byte_size: u64,
}

impl LanceScanHandle {
    async fn new(uri: &str) -> Result<Self> {
        let dataset = Dataset::open(uri).await?;
        let row_count = dataset.count_rows(None).await?;

        // Calculate total size from data files
        let path = uri.strip_prefix("file://").unwrap_or(uri);

        let byte_size = WalkDir::new(path)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
            .filter_map(|e| e.metadata().ok())
            .map(|m| m.len())
            .sum();

        Ok(Self {
            dataset,
            row_count,
            byte_size,
        })
    }
}

#[async_trait]
impl ScanHandle for LanceScanHandle {
    async fn scan(&self) -> Result<Vec<RecordBatch>> {
        let stream = self.dataset.scan().try_into_stream().await?;
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

/// Lance storage engine with configurable data storage version.
pub struct LanceEngine {
    runtime: Arc<Runtime>,
    version: LanceFileVersion,
    name: &'static str,
}

impl LanceEngine {
    /// Create a new Lance engine with the specified data storage version.
    pub fn with_version(version: LanceFileVersion, name: &'static str) -> Self {
        Self {
            runtime: Arc::new(
                tokio::runtime::Builder::new_current_thread()
                    .build()
                    .unwrap(),
            ),
            version,
            name,
        }
    }

    /// Create a Lance 2.0 engine (legacy format).
    pub fn v2_0() -> Self {
        Self::with_version(LanceFileVersion::V2_0, "lance-2.0")
    }

    /// Create a Lance 2.1 engine.
    pub fn v2_1() -> Self {
        Self::with_version(LanceFileVersion::V2_1, "lance-2.1")
    }

    /// Create a Lance 2.2 engine.
    pub fn v2_2() -> Self {
        Self::with_version(LanceFileVersion::V2_2, "lance-2.2")
    }

    /// Convert a URI to a Lance URI.
    fn to_lance_uri(&self, uri: &str) -> String {
        if uri.contains("://") {
            uri.to_string()
        } else {
            uri.to_string()
        }
    }

    /// Extract the file path from a URI for cache operations.
    fn uri_to_path<'a>(&self, uri: &'a str) -> &'a str {
        if let Some(path) = uri.strip_prefix("file://") {
            path
        } else if uri.contains("://") {
            uri
        } else {
            uri
        }
    }
}

impl ScanEngine for LanceEngine {
    fn name(&self) -> &'static str {
        self.name
    }

    fn runtime(&self) -> Arc<Runtime> {
        self.runtime.clone()
    }

    fn exists(&self, uri: &str) -> bool {
        self.runtime.block_on(async {
            let lance_uri = self.to_lance_uri(uri);
            Dataset::open(&lance_uri).await.is_ok()
        })
    }

    fn open(&self, uri: &str) -> Result<Arc<dyn ScanHandle>> {
        self.runtime.block_on(async {
            let lance_uri = self.to_lance_uri(uri);
            let handle = LanceScanHandle::new(&lance_uri).await?;
            Ok(Arc::new(handle) as Arc<dyn ScanHandle>)
        })
    }

    fn write(&self, uri: &str, batches: &[RecordBatch]) -> Result<Arc<dyn ScanHandle>> {
        self.runtime.block_on(async {
            let lance_uri = self.to_lance_uri(uri);

            // Get schema from first batch
            let schema = batches
                .first()
                .ok_or_else(|| anyhow::anyhow!("No batches to write"))?
                .schema();

            // Clone batches into owned iterator
            let batches_owned: Vec<RecordBatch> = batches.to_vec();
            let reader = RecordBatchIterator::new(batches_owned.into_iter().map(Ok), schema);

            let params = WriteParams {
                mode: WriteMode::Create,
                data_storage_version: Some(self.version),
                ..Default::default()
            };

            let _dataset = Dataset::write(reader, &lance_uri, Some(params)).await?;

            // Open the written dataset
            let handle = LanceScanHandle::new(&lance_uri).await?;
            Ok(Arc::new(handle) as Arc<dyn ScanHandle>)
        })
    }

    fn drop_cache(&self, uri: &str) -> Result<()> {
        let path = self.uri_to_path(uri);
        drop_directory_cache(Path::new(path))
    }
}
