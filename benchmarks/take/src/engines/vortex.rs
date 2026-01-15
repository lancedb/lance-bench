//! Vortex storage engine implementation.

use anyhow::Result;
use arrow::array::RecordBatch;
use async_trait::async_trait;
use indicatif::{ProgressBar, ProgressStyle};
use std::fs;
use std::path::Path;
use std::sync::Arc;
use tokio::runtime::Runtime;
use vortex::array::arrays::ChunkedArray;
use vortex::array::arrow::{FromArrowArray, IntoArrowArray};
use vortex::array::stream::ArrayStreamExt;
use vortex::array::{Array, ArrayRef};
use vortex::buffer::Buffer;
use vortex::dtype::DType;
use vortex::file::{OpenOptionsSessionExt, VortexFile, VortexWriteOptions};
use vortex::io::session::RuntimeSessionExt;
use vortex::scan::Selection;
use vortex::session::VortexSession;
use vortex::VortexSessionDefault;

use crate::cache::drop_directory_cache;
use crate::data::{create_schema, generate_vector_batch};
use crate::Config;

use super::traits::{DatasetHandle, Engine};

/// Handle to an open Vortex dataset.
pub struct VortexHandle {
    /// The opened Vortex file
    file: VortexFile,
}

impl VortexHandle {
    async fn new(path: &str, session: &VortexSession) -> Result<Self> {
        let file = session
            .open_options()
            .open(path)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to open Vortex file: {}", e))?;

        Ok(Self { file })
    }
}

#[async_trait]
impl DatasetHandle for VortexHandle {
    async fn take(&self, indices: &[u64]) -> Result<RecordBatch> {
        // Read all data from the file first
        let array = self
            .file
            .scan()
            .map_err(|e| anyhow::anyhow!("Failed to create scan: {}", e))?
            .with_selection(Selection::IncludeByIndex(Buffer::copy_from(indices)))
            .into_array_stream()
            .map_err(|e| anyhow::anyhow!("Failed to create array stream: {}", e))?
            .read_all()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to read array: {}", e))?;

        // Convert back to Arrow using the preferred conversion
        let arrow_array = array
            .into_arrow_preferred()
            .map_err(|e| anyhow::anyhow!("Failed to convert to Arrow: {}", e))?;

        // The result should be a struct array that we can convert to a RecordBatch
        let struct_array = arrow_array
            .as_any()
            .downcast_ref::<arrow::array::StructArray>()
            .ok_or_else(|| anyhow::anyhow!("Expected StructArray from Vortex"))?;

        let batch = RecordBatch::from(struct_array);
        Ok(batch)
    }
}

/// Vortex storage engine.
pub struct VortexEngine {
    session: VortexSession,
    runtime: Arc<Runtime>,
}

impl VortexEngine {
    pub fn new() -> Self {
        Self {
            session: VortexSession::default().with_tokio(),
            runtime: Arc::new(
                tokio::runtime::Builder::new_current_thread()
                    .build()
                    .unwrap(),
            ),
        }
    }
}

impl VortexEngine {
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

    /// Get the vortex file path within the dataset directory.
    fn get_vortex_file(&self, uri: &str) -> String {
        let base_path = self.uri_to_path(uri);
        format!("{}/data.vortex", base_path)
    }
}

#[async_trait]
impl Engine for VortexEngine {
    fn name(&self) -> &'static str {
        "vortex"
    }

    fn runtime(&self) -> Arc<Runtime> {
        self.runtime.clone()
    }

    fn exists(&self, uri: &str, expected_rows: usize) -> bool {
        self.runtime.block_on(async move {
            let vortex_file = self.get_vortex_file(uri);
            let path = Path::new(&vortex_file);

            if !path.exists() {
                println!("File does not exist");
                return false;
            }

            // Try to open and check row count
            println!("File exists, opening file");
            if let Ok(file) = self.session.open_options().open(vortex_file.as_str()).await {
                println!(
                    "File opened, checking row count {} == {}",
                    file.row_count() as usize,
                    expected_rows
                );
                return file.row_count() as usize == expected_rows;
            }
            println!("Error opening file");
            false
        })
    }

    fn open(&self, uri: &str) -> Result<Arc<dyn DatasetHandle>> {
        self.runtime.block_on(async {
            let vortex_file = self.get_vortex_file(uri);
            let handle = VortexHandle::new(&vortex_file, &self.session).await?;
            Ok(Arc::new(handle) as Arc<dyn DatasetHandle>)
        })
    }

    fn write(&self, uri: &str, config: &Config) -> Result<Arc<dyn DatasetHandle>> {
        self.runtime.block_on(async {
            let base_path = self.uri_to_path(uri);
            let vortex_file = self.get_vortex_file(uri);

            println!("\nGenerating dataset: {}", vortex_file);

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

            // Generate all batches and convert to Vortex arrays
            let mut vortex_chunks: Vec<ArrayRef> = Vec::with_capacity(num_batches);
            let mut vortex_dtype: Option<DType> = None;

            for _ in 0..num_batches {
                let batch = generate_vector_batch(
                    schema.clone(),
                    config.write_batch_size,
                    config.vector_dim,
                )?;

                // Convert Arrow RecordBatch to StructArray first, then to Vortex array
                let struct_array: arrow::array::StructArray = batch.into();
                let vortex_array = ArrayRef::from_arrow(&struct_array, false);

                if vortex_dtype.is_none() {
                    vortex_dtype = Some(vortex_array.dtype().clone());
                }

                vortex_chunks.push(vortex_array);
                pb.inc(1);
            }

            pb.finish();

            // Create a ChunkedArray from all the chunks
            let dtype = vortex_dtype.ok_or_else(|| anyhow::anyhow!("No batches generated"))?;
            let chunked = ChunkedArray::try_new(vortex_chunks, dtype)
                .map_err(|e| anyhow::anyhow!("Failed to create chunked array: {}", e))?;

            // Write to file
            let file = tokio::fs::File::create(&vortex_file).await?;
            VortexWriteOptions::new(self.session.clone())
                .write(file, chunked.to_array_stream())
                .await
                .map_err(|e| anyhow::anyhow!("Failed to write Vortex file: {}", e))?;

            // Open the written file
            let handle = VortexHandle::new(&vortex_file, &self.session).await?;
            Ok(Arc::new(handle) as Arc<dyn DatasetHandle>)
        })
    }

    fn drop_cache(&self, uri: &str) -> Result<()> {
        let path = self.uri_to_path(uri);
        drop_directory_cache(Path::new(path))
    }
}
