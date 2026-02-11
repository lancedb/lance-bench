//! Vortex storage engine implementation for scan benchmark.

use anyhow::Result;
use arrow::record_batch::RecordBatch;
use async_trait::async_trait;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use tokio::runtime::Runtime;
use vortex::array::arrays::ChunkedArray;
use vortex::array::arrow::{FromArrowArray, IntoArrowArray};
use vortex::array::stream::ArrayStreamExt;
use vortex::array::{Array, ArrayRef};
use vortex::dtype::DType;
use vortex::file::{OpenOptionsSessionExt, VortexFile, VortexWriteOptions};
use vortex::io::session::RuntimeSessionExt;
use vortex::session::VortexSession;
use vortex::VortexSessionDefault;

use crate::cache::drop_directory_cache;

use super::traits::{ScanEngine, ScanHandle};

/// Handle to an open Vortex file for scanning.
pub struct VortexScanHandle {
    /// The opened Vortex file
    file: VortexFile,
    /// File size in bytes
    byte_size: u64,
}

impl VortexScanHandle {
    async fn new(path: &str, session: &VortexSession) -> Result<Self> {
        let file = session
            .open_options()
            .open(path)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to open Vortex file: {}", e))?;

        // Get file size
        let byte_size = fs::metadata(path).map(|m| m.len()).unwrap_or(0);

        Ok(Self { file, byte_size })
    }
}

#[async_trait]
impl ScanHandle for VortexScanHandle {
    async fn scan(&self) -> Result<Vec<RecordBatch>> {
        let array = self
            .file
            .scan()
            .map_err(|e| anyhow::anyhow!("Failed to create scan: {}", e))?
            .into_array_stream()
            .map_err(|e| anyhow::anyhow!("Failed to create array stream: {}", e))?
            .read_all()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to read array: {}", e))?;

        // Convert back to Arrow
        let arrow_array = array
            .into_arrow_preferred()
            .map_err(|e| anyhow::anyhow!("Failed to convert to Arrow: {}", e))?;

        // The result should be a struct array that we can convert to a RecordBatch
        let struct_array = arrow_array
            .as_any()
            .downcast_ref::<arrow::array::StructArray>()
            .ok_or_else(|| anyhow::anyhow!("Expected StructArray from Vortex"))?;

        let batch = RecordBatch::from(struct_array);
        Ok(vec![batch])
    }

    fn row_count(&self) -> usize {
        self.file.row_count() as usize
    }

    fn byte_size(&self) -> u64 {
        self.byte_size
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

impl Default for VortexEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl ScanEngine for VortexEngine {
    fn name(&self) -> &'static str {
        "vortex"
    }

    fn runtime(&self) -> Arc<Runtime> {
        self.runtime.clone()
    }

    fn exists(&self, uri: &str) -> bool {
        let vortex_file = self.get_vortex_file(uri);
        Path::new(&vortex_file).exists()
    }

    fn open(&self, uri: &str) -> Result<Arc<dyn ScanHandle>> {
        self.runtime.block_on(async {
            let vortex_file = self.get_vortex_file(uri);
            let handle = VortexScanHandle::new(&vortex_file, &self.session).await?;
            Ok(Arc::new(handle) as Arc<dyn ScanHandle>)
        })
    }

    fn write(&self, uri: &str, batches: &[RecordBatch]) -> Result<Arc<dyn ScanHandle>> {
        self.runtime.block_on(async {
            let base_path = self.uri_to_path(uri);
            let vortex_file = self.get_vortex_file(uri);

            // Create the directory
            fs::create_dir_all(base_path)?;

            // Convert all batches to Vortex arrays
            let mut vortex_chunks: Vec<ArrayRef> = Vec::with_capacity(batches.len());
            let mut vortex_dtype: Option<DType> = None;

            for batch in batches {
                // Convert Arrow RecordBatch to StructArray first, then to Vortex array
                let struct_array: arrow::array::StructArray = batch.clone().into();
                let vortex_array = ArrayRef::from_arrow(&struct_array, false);

                if vortex_dtype.is_none() {
                    vortex_dtype = Some(vortex_array.dtype().clone());
                }

                vortex_chunks.push(vortex_array);
            }

            // Create a ChunkedArray from all the chunks
            let dtype = vortex_dtype.ok_or_else(|| anyhow::anyhow!("No batches to write"))?;
            let chunked = ChunkedArray::try_new(vortex_chunks, dtype)
                .map_err(|e| anyhow::anyhow!("Failed to create chunked array: {}", e))?;

            // Write to file
            let file = tokio::fs::File::create(&vortex_file).await?;
            VortexWriteOptions::new(self.session.clone())
                .write(file, chunked.to_array_stream())
                .await
                .map_err(|e| anyhow::anyhow!("Failed to write Vortex file: {}", e))?;

            // Open the written file
            let handle = VortexScanHandle::new(&vortex_file, &self.session).await?;
            Ok(Arc::new(handle) as Arc<dyn ScanHandle>)
        })
    }

    fn drop_cache(&self, uri: &str) -> Result<()> {
        let path = self.uri_to_path(uri);
        drop_directory_cache(Path::new(path))
    }
}
