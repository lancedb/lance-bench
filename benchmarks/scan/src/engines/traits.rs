//! Engine trait definitions for scan benchmark.

use anyhow::Result;
use arrow::record_batch::RecordBatch;
use async_trait::async_trait;
use std::sync::Arc;
use tokio::runtime::Runtime;

/// A handle to an open dataset that can execute scans.
#[async_trait]
pub trait ScanHandle: Send + Sync {
    /// Scan and return all rows from the dataset.
    async fn scan(&self) -> Result<Vec<RecordBatch>>;

    /// Returns the total row count in this dataset.
    fn row_count(&self) -> usize;

    /// Returns the byte size of the dataset on disk.
    fn byte_size(&self) -> u64;
}

/// Engine trait for different storage backends.
pub trait ScanEngine: Send + Sync {
    /// Returns the name of this engine.
    fn name(&self) -> &'static str;

    /// Get the runtime for the engine.
    fn runtime(&self) -> Arc<Runtime>;

    /// Check if a dataset exists at the given URI.
    fn exists(&self, uri: &str) -> bool;

    /// Open an existing dataset.
    fn open(&self, uri: &str) -> Result<Arc<dyn ScanHandle>>;

    /// Write batches to a new dataset, returning a handle to the written data.
    fn write(&self, uri: &str, batches: &[RecordBatch]) -> Result<Arc<dyn ScanHandle>>;

    /// Drop the dataset from the kernel page cache.
    fn drop_cache(&self, uri: &str) -> Result<()>;
}

/// Registry of available engines.
pub struct EngineRegistry {
    engines: Vec<Arc<dyn ScanEngine>>,
}

impl EngineRegistry {
    pub fn new() -> Self {
        Self {
            engines: Vec::new(),
        }
    }

    pub fn register(&mut self, engine: Arc<dyn ScanEngine>) {
        self.engines.push(engine);
    }

    pub fn get(&self, name: &str) -> Option<Arc<dyn ScanEngine>> {
        self.engines.iter().find(|e| e.name() == name).cloned()
    }

    pub fn available(&self) -> Vec<&'static str> {
        self.engines.iter().map(|e| e.name()).collect()
    }

    pub fn all(&self) -> Vec<Arc<dyn ScanEngine>> {
        self.engines.clone()
    }
}

impl Default for EngineRegistry {
    fn default() -> Self {
        Self::new()
    }
}
