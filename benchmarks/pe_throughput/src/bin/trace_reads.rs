//! Standalone experiment: open the dataset, do a few takes, and log every
//! object-store read with the requested range vs the actual file size.

use std::fmt;
use std::ops::Range;
use std::sync::Arc;

use bytes::Bytes;
use futures::stream::BoxStream;
use lance::dataset::{Dataset, ProjectionRequest};
use object_store::{
    path::Path as OSPath, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta,
    ObjectStore, PutOptions, PutPayload, PutResult, Result as OSResult,
};
use rand::Rng;

// ---------------------------------------------------------------------------
// Logging object store wrapper
// ---------------------------------------------------------------------------

struct LoggingObjectStore {
    inner: Arc<dyn ObjectStore>,
    prefix: String,
}

impl fmt::Debug for LoggingObjectStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LoggingObjectStore").finish()
    }
}

impl fmt::Display for LoggingObjectStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LoggingObjectStore({})", self.inner)
    }
}

#[async_trait::async_trait]
impl ObjectStore for LoggingObjectStore {
    async fn put(&self, location: &OSPath, payload: PutPayload) -> OSResult<PutResult> {
        self.inner.put(location, payload).await
    }
    async fn put_opts(&self, location: &OSPath, bytes: PutPayload, opts: PutOptions) -> OSResult<PutResult> {
        self.inner.put_opts(location, bytes, opts).await
    }
    async fn put_multipart(&self, location: &OSPath) -> OSResult<Box<dyn MultipartUpload>> {
        self.inner.put_multipart(location).await
    }
    async fn put_multipart_opts(&self, location: &OSPath, opts: object_store::PutMultipartOptions) -> OSResult<Box<dyn MultipartUpload>> {
        self.inner.put_multipart_opts(location, opts).await
    }

    async fn get(&self, location: &OSPath) -> OSResult<GetResult> {
        println!("[TRACE] get({}/{})", self.prefix, location);
        self.inner.get(location).await
    }

    async fn get_opts(&self, location: &OSPath, opts: GetOptions) -> OSResult<GetResult> {
        println!("[TRACE] get_opts({}/{}, range={:?})", self.prefix, location, opts.range);
        let result = self.inner.get_opts(location, opts).await;
        if let Ok(ref r) = result {
            println!("[TRACE]   -> returned range {}..{}", r.range.start, r.range.end);
        }
        result
    }

    async fn get_range(&self, location: &OSPath, range: Range<u64>) -> OSResult<Bytes> {
        println!("[TRACE] get_range({}/{}, {}..{})", self.prefix, location, range.start, range.end);
        let result = self.inner.get_range(location, range.clone()).await;
        if let Ok(ref b) = result {
            println!("[TRACE]   -> returned {} bytes", b.len());
        }
        result
    }

    async fn get_ranges(&self, location: &OSPath, ranges: &[Range<u64>]) -> OSResult<Vec<Bytes>> {
        println!("[TRACE] get_ranges({}/{}, {} ranges)", self.prefix, location, ranges.len());
        for r in ranges {
            println!("[TRACE]   range: {}..{}", r.start, r.end);
        }
        self.inner.get_ranges(location, ranges).await
    }

    async fn head(&self, location: &OSPath) -> OSResult<ObjectMeta> {
        println!("[TRACE] head({}/{})", self.prefix, location);
        let meta = self.inner.head(location).await;
        if let Ok(ref m) = meta {
            println!("[TRACE]   -> size={}", m.size);
        }
        meta
    }

    async fn delete(&self, location: &OSPath) -> OSResult<()> {
        self.inner.delete(location).await
    }
    fn delete_stream<'a>(&'a self, locations: BoxStream<'a, OSResult<OSPath>>) -> BoxStream<'a, OSResult<OSPath>> {
        self.inner.delete_stream(locations)
    }
    fn list(&self, prefix: Option<&OSPath>) -> BoxStream<'static, OSResult<ObjectMeta>> {
        self.inner.list(prefix)
    }
    fn list_with_offset(&self, prefix: Option<&OSPath>, offset: &OSPath) -> BoxStream<'static, OSResult<ObjectMeta>> {
        self.inner.list_with_offset(prefix, offset)
    }
    async fn list_with_delimiter(&self, prefix: Option<&OSPath>) -> OSResult<ListResult> {
        self.inner.list_with_delimiter(prefix).await
    }
    async fn copy(&self, from: &OSPath, to: &OSPath) -> OSResult<()> {
        self.inner.copy(from, to).await
    }
    async fn rename(&self, from: &OSPath, to: &OSPath) -> OSResult<()> {
        self.inner.rename(from, to).await
    }
    async fn copy_if_not_exists(&self, from: &OSPath, to: &OSPath) -> OSResult<()> {
        self.inner.copy_if_not_exists(from, to).await
    }
    async fn rename_if_not_exists(&self, from: &OSPath, to: &OSPath) -> OSResult<()> {
        self.inner.rename_if_not_exists(from, to).await
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let dataset_uri = "s3://weston-s3-lance-test/pe-throughput";

    // Open dataset normally, then swap in the logging wrapper
    let ds = Dataset::open(dataset_uri).await?;
    let mut store = ds.object_store().clone();
    let prefix = store.store_prefix.clone();
    store.inner = Arc::new(LoggingObjectStore {
        inner: store.inner.clone(),
        prefix: prefix.clone(),
    });
    let ds = ds.with_object_store(Arc::new(store), None);

    let num_fragments = ds.get_fragments().len() as u32;
    let rows_per_fragment = 16384u32;
    println!("Dataset opened: {} fragments", num_fragments);

    // Print file sizes for reference
    let frags = ds.get_fragments();
    for (i, frag) in frags.iter().take(3).enumerate() {
        for df in &frag.metadata().files {
            println!("Fragment {} file: size={:?}", i, df.file_size_bytes);
        }
    }

    // Get field IDs for all columns
    let schema = ds.schema();
    let all_columns: Vec<(String, String)> = vec![
        "id", "vector",
        "str_payload_0", "str_payload_1", "str_payload_2", "str_payload_3",
        "float_payload_0", "float_payload_1", "float_payload_2", "float_payload_3",
    ].into_iter().map(|c| (c.to_string(), c.to_string())).collect();

    // Do a single take of 5 random rows and log all I/O
    let mut rng = rand::thread_rng();
    let indices: Vec<u64> = (0..5)
        .map(|_| rng.gen_range(0..num_fragments as u64 * rows_per_fragment as u64))
        .collect();

    println!("\n=== Performing take of {} rows ===", indices.len());
    println!("Row indices: {:?}", indices);
    println!();

    let _batch = ds
        .take(
            &indices,
            ProjectionRequest::Sql(all_columns),
        )
        .await?;

    println!("\n=== Take completed, {} rows returned ===", _batch.num_rows());

    Ok(())
}
