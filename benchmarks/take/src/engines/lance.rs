//! Lance storage engine implementation.

use anyhow::Result;
use arrow::array::RecordBatchIterator;
use arrow::record_batch::RecordBatch;
use async_trait::async_trait;
use indicatif::{ProgressBar, ProgressStyle};
use lance::dataset::{Dataset, WriteMode, WriteParams};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::runtime::Runtime;

use crate::cache::drop_directory_cache;
use crate::data::{create_schema, generate_vector_batch};
use crate::Config;

use super::traits::{DatasetHandle, Engine};

/// Handle to an open Lance dataset.
pub struct LanceHandle {
    dataset: Dataset,
}

#[async_trait]
impl DatasetHandle for LanceHandle {
    async fn take(&self, indices: &[u64]) -> Result<RecordBatch> {
        Ok(self
            .dataset
            .take(
                indices,
                lance::dataset::ProjectionRequest::Sql(vec![(
                    "vector".to_string(),
                    "vector".to_string(),
                )]),
            )
            .await?)
    }
}

/// Lance storage engine.
pub struct LanceEngine {
    runtime: Arc<Runtime>,
}

impl LanceEngine {
    pub fn new() -> Self {
        Self {
            runtime: Arc::new(
                tokio::runtime::Builder::new_current_thread()
                    .build()
                    .unwrap(),
            ),
        }
    }

    /// Convert a URI to a Lance URI with uring support.
    /// If already has a scheme, use as-is; otherwise prepend file+uring://
    fn to_lance_uri(&self, uri: &str) -> String {
        if uri.contains("://") {
            uri.to_string()
        } else {
            format!("file+uring://{}", uri)
        }
    }

    /// Extract the file path from a URI for cache operations.
    fn uri_to_path<'a>(&self, uri: &'a str) -> &'a str {
        if let Some(path) = uri.strip_prefix("file+uring://") {
            path
        } else if let Some(path) = uri.strip_prefix("file://") {
            path
        } else if uri.contains("://") {
            // For other schemes (s3://, etc.), return as-is
            uri
        } else {
            // Plain path
            uri
        }
    }
}

impl Default for LanceEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Engine for LanceEngine {
    fn name(&self) -> &'static str {
        "lance"
    }

    fn runtime(&self) -> Arc<Runtime> {
        self.runtime.clone()
    }

    fn exists(&self, uri: &str, expected_rows: usize) -> bool {
        self.runtime.block_on(async {
            let lance_uri = self.to_lance_uri(uri);
            if let Ok(dataset) = Dataset::open(&lance_uri).await {
                if let Ok(count) = dataset.count_rows(None).await {
                    return count == expected_rows;
                }
            }
            false
        })
    }

    fn open(&self, uri: &str) -> Result<Arc<dyn DatasetHandle>> {
        self.runtime.block_on(async {
            let lance_uri = self.to_lance_uri(uri);
            let dataset = Dataset::open(&lance_uri).await?;
            Ok(Arc::new(LanceHandle { dataset }) as Arc<dyn DatasetHandle>)
        })
    }

    fn write(&self, uri: &str, config: &Config) -> Result<Arc<dyn DatasetHandle>> {
        self.runtime.block_on(async {
            let lance_uri = self.to_lance_uri(uri);
            println!("\nGenerating dataset: {}", lance_uri);

            let num_batches = config.rows_per_dataset / config.write_batch_size;
            let pb = ProgressBar::new(num_batches as u64);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("  Writing batches [{bar:40}] {pos}/{len}")
                    .unwrap(),
            );

            let schema = create_schema(config.vector_dim);
            let batch_size = config.write_batch_size;
            let dim = config.vector_dim;

            // Use atomic counter for progress tracking
            let counter = Arc::new(AtomicU64::new(0));
            let counter_clone = counter.clone();

            let batches = (0..num_batches).map(move |_| {
                let batch = generate_vector_batch(schema.clone(), batch_size, dim);
                let count = counter_clone.fetch_add(1, Ordering::Relaxed);
                pb.set_position(count + 1);
                batch
            });

            let reader = RecordBatchIterator::new(batches, create_schema(config.vector_dim));

            let params = WriteParams {
                mode: WriteMode::Create,
                max_rows_per_file: config.rows_per_dataset,
                ..Default::default()
            };

            let dataset = Dataset::write(reader, &lance_uri, Some(params)).await?;

            Ok(Arc::new(LanceHandle { dataset }) as Arc<dyn DatasetHandle>)
        })
    }

    fn drop_cache(&self, uri: &str) -> Result<()> {
        let path = self.uri_to_path(uri);
        drop_directory_cache(Path::new(path))
    }
}
