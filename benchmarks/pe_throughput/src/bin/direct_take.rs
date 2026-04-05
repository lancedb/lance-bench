//! Direct take test — bypasses the PE, uses lance Dataset directly with a
//! ReadThroughPageCache wrapping the object store.  Tests whether the
//! concurrency bug is in the PE wrapping or in the page cache itself.

use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use lance::dataset::{Dataset, ProjectionRequest};
use rand::Rng;
use tokio::sync::Semaphore;

use sophon_caching::object_store::config::{
    ChainedCachesConfig, ObjectStoreCacheConfig,
};
use sophon_caching::object_store::config::{CacheRegistry, PrewarmRegistry};
use lance::io::WrappingObjectStore;

#[tokio::main]
async fn main() -> Result<()> {
    let dataset_uri = "s3://weston-s3-lance-test/pe-throughput";
    let num_fragments = 1024u32;
    let rows_per_fragment = 16384u32;
    let take_size = 100;
    let num_takes = 200;
    let concurrency = std::env::var("CONCURRENCY").unwrap_or("2".into()).parse::<usize>().unwrap();

    // Open dataset
    let ds = Dataset::open(dataset_uri).await?;
    println!("Dataset opened: {} fragments", ds.get_fragments().len());

    // Build disk page cache
    let cache_dir = tempfile::tempdir()?;
    let disk_cache_dir = cache_dir.path().join("disk_cache");
    std::fs::create_dir_all(&disk_cache_dir)?;

    let cache_config = ChainedCachesConfig::new(vec![
        ObjectStoreCacheConfig::disk(
            [disk_cache_dir.as_path()],
            256 * 1024 * 1024, // 256 MB
            Some(64 * 1024),   // 64 KB pages
            Some(2),
            None,
            None,
            None,
        ),
    ]);

    let mut cache_registry = CacheRegistry {
        disk_caches: Vec::new(),
        prewarms: Arc::new(PrewarmRegistry::new(20, 3)),
    };
    let read_through_cache = cache_config
        .make_read_through_cache_chain(&mut cache_registry, None)
        .await?;

    // Wrap the dataset's object store with the page cache
    let mut store = ds.object_store().clone();
    let prefix = store.store_prefix.clone();
    if let Some(cache) = &read_through_cache {
        store.inner = cache.wrap(&prefix, store.inner.clone());
    }
    let ds = Arc::new(ds.with_object_store(Arc::new(store), None));

    let all_columns: Vec<(String, String)> = vec![
        "id", "vector",
        "str_payload_0", "str_payload_1", "str_payload_2", "str_payload_3",
        "float_payload_0", "float_payload_1", "float_payload_2", "float_payload_3",
    ].into_iter().map(|c| (c.to_string(), c.to_string())).collect();

    // Run takes
    let sem = Arc::new(Semaphore::new(concurrency));
    let mut handles = Vec::new();

    println!("Running {} takes at concurrency={} ...", num_takes, concurrency);
    let wall_start = Instant::now();

    for i in 0..num_takes {
        let sem = sem.clone();
        let ds = ds.clone();
        let projection = all_columns.clone();

        let mut rng = rand::thread_rng();
        let indices: Vec<u64> = (0..take_size)
            .map(|_| rng.gen_range(0..num_fragments as u64 * rows_per_fragment as u64))
            .collect();

        handles.push(tokio::spawn(async move {
            let _permit = sem.acquire().await.unwrap();
            let result = ds
                .take(&indices, ProjectionRequest::Sql(projection))
                .await;
            match result {
                Ok(batch) => {
                    if i % 50 == 0 {
                        println!("  take {}: {} rows OK", i, batch.num_rows());
                    }
                }
                Err(e) => {
                    eprintln!("  take {} FAILED: {}", i, e);
                }
            }
        }));
    }

    for h in handles {
        h.await?;
    }

    let elapsed = wall_start.elapsed();
    println!("Done: {} takes in {:.2}s ({:.2} takes/sec)",
        num_takes, elapsed.as_secs_f64(),
        num_takes as f64 / elapsed.as_secs_f64());

    Ok(())
}
