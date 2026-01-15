//! Take Benchmark
//!
//! Benchmarks take (point lookup) performance across different storage engines.
//!
//! Supports:
//! - Lance (default)
//! - Parquet (future)
//! - Vortex (future)

use anyhow::Result;
use clap::Parser;
use crossbeam_channel::{bounded, Receiver, Sender};
use futures::stream::{self, StreamExt};
use indicatif::{ProgressBar, ProgressStyle};
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use std::time::Instant;
use tokio::runtime::Runtime;

mod cache;
mod data;
mod engines;
mod stats;

use engines::{create_registry, DatasetHandle};
use stats::compute_statistics;

extern crate jemallocator;

#[global_allocator]
static GLOBAL: jemallocator::Jemalloc = jemallocator::Jemalloc;

/// Take benchmark configuration.
#[derive(Parser, Debug, Clone)]
#[command(name = "take-benchmark")]
#[command(about = "Benchmark take (point lookup) performance across storage engines")]
pub struct Config {
    /// Storage engine to use
    #[arg(short, long, default_value = "lance")]
    pub engine: String,

    /// Number of rows per dataset
    #[arg(long, default_value_t = 1_000_000)]
    pub rows_per_dataset: usize,

    /// Batch size when writing data
    #[arg(long, default_value_t = 100_000)]
    pub write_batch_size: usize,

    /// Vector dimension
    #[arg(long, default_value_t = 768)]
    pub vector_dim: usize,

    /// Number of queries to execute
    #[arg(long, default_value_t = 2_000)]
    pub num_queries: usize,

    /// Number of rows per query
    #[arg(long, default_value_t = 500)]
    pub rows_per_query: usize,

    /// Number of worker runtimes
    #[arg(long, default_value_t = 16)]
    pub num_runtimes: usize,

    /// Concurrent queries per runtime
    #[arg(long, default_value_t = 4)]
    pub concurrent_queries: usize,

    /// Dataset URIs (can be specified multiple times)
    #[arg(short, long, default_value = "file:///tmp/dataset")]
    pub dataset_uri: Vec<String>,

    /// Skip warmup phase
    #[arg(long, default_value_t = false)]
    pub skip_warmup: bool,

    /// Skip cache drop between warmup and timed phase
    #[arg(long, default_value_t = false)]
    pub skip_cache_drop: bool,
}

static ROW_COUNTER: AtomicUsize = AtomicUsize::new(0);

// Query task: (dataset_idx, query_indices)
type QueryTask = (usize, Vec<u64>);

async fn execute_query(dataset: Arc<dyn DatasetHandle>, query_indices: Vec<u64>) -> Result<f64> {
    let start = Instant::now();

    let batch = dataset.take(&query_indices).await?;

    ROW_COUNTER.fetch_add(batch.num_rows(), std::sync::atomic::Ordering::Relaxed);

    Ok(start.elapsed().as_secs_f64())
}

fn run_queries(
    datasets: Vec<Arc<dyn DatasetHandle>>,
    queries: Vec<Vec<u64>>,
    warmup: bool,
    config: &Config,
    runtime: Arc<Runtime>,
) -> Result<Vec<f64>> {
    let desc = if warmup {
        "Warmup queries"
    } else {
        "Timed queries"
    };
    let pb = ProgressBar::new(queries.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(&format!("  {} [{{bar:40}}] {{pos}}/{{len}}", desc))
            .unwrap(),
    );

    let num_datasets = datasets.len();
    let num_runtimes = config.num_runtimes;
    let concurrent_queries = config.concurrent_queries;

    // Create MPMC channel for query tasks
    let (tx, rx): (Sender<QueryTask>, Receiver<QueryTask>) = bounded(queries.len());

    // Send all queries to the channel
    for (i, query) in queries.into_iter().enumerate() {
        let dataset_idx = i % num_datasets;
        tx.send((dataset_idx, query))?;
    }
    drop(tx); // Close the sender so threads know when to stop

    // Spawn worker threads
    let mut handles = Vec::new();
    let latencies = Arc::new(std::sync::Mutex::new(Vec::new()));

    for thread_idx in 0..num_runtimes {
        let rx = rx.clone();
        let datasets = datasets.clone();
        let pb = pb.clone();
        let latencies = latencies.clone();

        let runtime = runtime.clone();

        let handle = std::thread::spawn(move || {
            runtime.block_on(async move {
                // Process queries from the queue with concurrency control
                let query_stream = stream::iter(std::iter::from_fn(|| rx.recv().ok()))
                    .map(|(dataset_idx, query)| {
                        let dataset = datasets[dataset_idx].clone();
                        let pb = pb.clone();
                        let latencies = latencies.clone();

                        tokio::task::spawn(async move {
                            let result = execute_query(dataset, query).await;
                            pb.inc(1);

                            let latency = result.unwrap_or_else(|e| {
                                eprintln!("Query failed in thread {}: {:?}", thread_idx, e);
                                0.0f64
                            });

                            if !warmup {
                                latencies.lock().unwrap().push(latency);
                            }
                        })
                    })
                    .buffer_unordered(concurrent_queries);

                // Collect all results
                query_stream
                    .for_each(|result| async {
                        if let Err(e) = result {
                            eprintln!("Query failed in thread {}: {:?}", thread_idx, e);
                        }
                    })
                    .await;
            });
        });

        handles.push(handle);
    }

    // Wait for all threads to complete
    for handle in handles {
        handle
            .join()
            .map_err(|_| anyhow::anyhow!("Thread panicked"))?;
    }

    pb.finish();

    let latencies = Arc::try_unwrap(latencies).unwrap().into_inner().unwrap();

    Ok(latencies)
}

fn main() -> Result<()> {
    env_logger::init();

    let config = Config::parse();

    // Get the engine
    let registry = create_registry();
    let engine = registry.get(&config.engine).ok_or_else(|| {
        anyhow::anyhow!(
            "Unknown engine '{}'. Available engines: {:?}",
            config.engine,
            registry.available()
        )
    })?;

    // Build dataset URIs with engine as child folder
    // e.g., /tmp/dataset -> /tmp/dataset/lance
    let dataset_uris: Vec<String> = config
        .dataset_uri
        .iter()
        .map(|uri| {
            let uri = uri.trim_end_matches('/');
            format!("{}/{}", uri, engine.name())
        })
        .collect();

    println!("{}", "=".repeat(60));
    println!("Take Benchmark");
    println!("{}", "=".repeat(60));
    println!("\nConfiguration:");
    println!("  Engine: {}", engine.name());
    println!("  Datasets: {}", dataset_uris.len());
    println!("  Vector dimensions: {}", config.vector_dim);
    println!("  Rows per dataset: {}", config.rows_per_dataset);
    println!("  Num queries: {}", config.num_queries);
    println!("  Rows per query: {}", config.rows_per_query);
    println!("  Number of runtimes: {}", config.num_runtimes);
    println!(
        "  Concurrent queries per runtime: {}",
        config.concurrent_queries
    );

    // Step 1: Create datasets
    println!("\n{}", "=".repeat(60));
    println!("Step 1: Loading/Creating Datasets");
    println!("{}", "=".repeat(60));

    let engine = registry.get(&config.engine).ok_or_else(|| {
        anyhow::anyhow!(
            "Unknown engine '{}'. Available engines: {:?}",
            config.engine,
            registry.available()
        )
    })?;

    let mut datasets: Vec<Arc<dyn DatasetHandle>> = Vec::new();
    for (i, uri) in dataset_uris.iter().enumerate() {
        println!("\nDataset {}/{}: {}", i + 1, dataset_uris.len(), uri);

        println!("Checking for existence of dataset...");
        let dataset = if engine.exists(uri, config.rows_per_dataset) {
            println!(
                "  Dataset exists with {} rows - loading",
                config.rows_per_dataset
            );
            engine.open(uri)?
        } else {
            println!("  Dataset not found or has wrong row count - creating");
            engine.write(uri, &config)?
        };

        datasets.push(dataset);
    }

    // Step 2: Generate queries
    println!("\n{}", "=".repeat(60));
    println!("Step 2: Generating Queries");
    println!("{}", "=".repeat(60));
    println!("\nGenerating {} query indices...", config.num_queries);
    let start = Instant::now();
    let queries = data::generate_queries(
        config.num_queries,
        config.rows_per_query,
        config.rows_per_dataset,
    );
    let elapsed = start.elapsed();
    println!("  Done in {:.2}s", elapsed.as_secs_f64());

    // Step 3: Warmup phase
    if !config.skip_warmup {
        println!("\n{}", "=".repeat(60));
        println!("Step 3: Warmup Phase");
        println!("{}", "=".repeat(60));
        println!("\nExecuting {} queries...", config.num_queries);
        run_queries(
            datasets.clone(),
            queries.clone(),
            true,
            &config,
            engine.runtime(),
        )?;
    }

    // Step 4: Drop cache
    if !config.skip_cache_drop {
        println!("\n{}", "=".repeat(60));
        println!("Step 4: Dropping Page Cache");
        println!("{}", "=".repeat(60));
        println!("\nDropping dataset files from kernel page cache...");
        for (i, uri) in dataset_uris.iter().enumerate() {
            println!("\n  Dataset {}/{}: {}", i + 1, dataset_uris.len(), uri);
            engine.drop_cache(uri)?;
        }
    }

    // Step 5: Timed phase
    println!("\n{}", "=".repeat(60));
    println!("Step 5: Timed Phase");
    println!("{}", "=".repeat(60));
    println!("\nExecuting {} queries...", config.num_queries);
    let start = Instant::now();
    let latencies = run_queries(datasets, queries, false, &config, engine.runtime())?;
    let elapsed = start.elapsed();

    // Step 6: Compute and display results
    println!("\n{}", "=".repeat(60));
    println!("BENCHMARK RESULTS");
    println!("{}", "=".repeat(60));

    let stats = compute_statistics(&latencies);
    let throughput = config.num_queries as f64 / elapsed.as_secs_f64();

    println!("\nLatency Statistics (seconds):");
    println!("  Mean:   {:.6}", stats.mean);
    println!("  Std:    {:.6}", stats.std);
    println!("  Min:    {:.6}", stats.min);
    println!("  Max:    {:.6}", stats.max);
    println!("  p50:    {:.6}", stats.p50);
    println!("  p95:    {:.6}", stats.p95);
    println!("  p99:    {:.6}", stats.p99);

    println!("\nThroughput: {:.2} queries/sec", throughput);

    println!("\n{}", "=".repeat(60));
    println!("Benchmark Complete!");
    println!("{}", "=".repeat(60));

    println!(
        "  Total rows scanned: {}",
        ROW_COUNTER.load(std::sync::atomic::Ordering::Relaxed)
    );

    Ok(())
}
