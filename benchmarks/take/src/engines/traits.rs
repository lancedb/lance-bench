//! Engine trait definition for benchmark engines.

use anyhow::Result;
use arrow::record_batch::RecordBatch;
use async_trait::async_trait;
use std::sync::Arc;
use tokio::runtime::Runtime;

use crate::Config;

/// A handle to an open dataset that can execute queries.
#[async_trait]
pub trait DatasetHandle: Send + Sync {
    /// Execute a take query, returning the specified row indices.
    async fn take(&self, indices: &[u64]) -> Result<RecordBatch>;
}

/// Engine trait for different storage backends.
#[async_trait]
pub trait Engine: Send + Sync {
    /// Returns the name of this engine.
    fn name(&self) -> &'static str;

    /// Get the runtime for the engine.
    fn runtime(&self) -> Arc<Runtime>;

    /// Check if a dataset exists at the given URI with the expected row count.
    fn exists(&self, uri: &str, expected_rows: usize) -> bool;

    /// Open an existing dataset.
    fn open(&self, uri: &str) -> Result<Arc<dyn DatasetHandle>>;

    /// Write data to a new dataset.
    fn write(&self, uri: &str, config: &Config) -> Result<Arc<dyn DatasetHandle>>;

    /// Drop the dataset from the kernel page cache.
    fn drop_cache(&self, uri: &str) -> Result<()>;
}

/// Registry of available engines.
pub struct EngineRegistry {
    engines: Vec<Arc<dyn Engine>>,
}

impl EngineRegistry {
    pub fn new() -> Self {
        Self {
            engines: Vec::new(),
        }
    }

    pub fn register(&mut self, engine: Arc<dyn Engine>) {
        self.engines.push(engine);
    }

    pub fn get(&self, name: &str) -> Option<Arc<dyn Engine>> {
        self.engines.iter().find(|e| e.name() == name).cloned()
    }

    pub fn available(&self) -> Vec<&'static str> {
        self.engines.iter().map(|e| e.name()).collect()
    }
}

impl Default for EngineRegistry {
    fn default() -> Self {
        Self::new()
    }
}
