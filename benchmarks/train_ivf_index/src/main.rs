//! IVF/PQ Index Training Benchmark
//!
//! Benchmarks how long it takes (and how much RAM / disk it uses) to train an
//! IVF/PQ vector index on Lance datasets with randomly generated vectors.
//!
//! Parameterized on:
//! - Number of vectors
//! - Number of dimensions
//! - Number of IVF partitions
//! - Distance type (L2, Cosine, Dot)
//!
//! The number of PQ sub-vectors is always dimension / 16.

use anyhow::{Context, Result};
use arrow_array::types::Float32Type;
use clap::Parser;
use lance::Dataset;
use lance_datagen::{array, gen_batch, BatchCount, Dimension, RowCount};
use lance_index::progress::IndexBuildProgress;
use lance_index::{DatasetIndexExt, IndexType};
use lance_linalg::distance::DistanceType;
use serde::Serialize;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tokio_metrics::TaskMonitor;

mod memory;
use memory::{get_cpu_time_secs, get_rss_bytes, PeakRssMonitor};

/// Recursively compute the total size of all files under a directory.
fn get_dir_size_bytes(path: &Path) -> u64 {
    walkdir(path).unwrap_or(0)
}

fn entry_size(entry: std::io::Result<std::fs::DirEntry>) -> std::io::Result<u64> {
    let entry = entry?;
    let ft = entry.file_type()?;
    if ft.is_file() {
        Ok(entry.metadata()?.len())
    } else if ft.is_dir() {
        walkdir(&entry.path())
    } else {
        Ok(0)
    }
}

fn walkdir(path: &Path) -> std::io::Result<u64> {
    let mut total = 0u64;
    if path.is_dir() {
        for entry in std::fs::read_dir(path)? {
            total += entry_size(entry).unwrap_or(0);
        }
    }
    Ok(total)
}

extern crate jemallocator;

#[global_allocator]
static GLOBAL: jemallocator::Jemalloc = jemallocator::Jemalloc;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser, Debug)]
#[command(name = "train-ivf-index-benchmark")]
#[command(about = "Benchmark IVF/PQ vector index training on Lance datasets")]
struct Args {
    /// Number of vectors in the dataset
    #[arg(long, default_value = "100000000")]
    num_vectors: usize,

    /// Dimensionality of each vector
    #[arg(long, default_value = "768")]
    dimensions: usize,

    /// Number of IVF partitions (default: min(num_vectors/4096, sqrt(num_vectors)))
    #[arg(long)]
    num_partitions: Option<usize>,

    /// Distance metric: l2, cosine, dot, or all
    #[arg(long, default_value = "l2")]
    distance_type: String,

    /// JSON output path
    #[arg(long, default_value = "ivf-index-results.json")]
    output: PathBuf,

    /// Cache directory for generated Lance datasets
    #[arg(long, default_value_os_t = default_cache_dir())]
    cache_dir: PathBuf,

    /// Force re-creation of the Lance dataset
    #[arg(long)]
    force_recreate: bool,

    /// Path for CSV output of per-callback RSS/CPU samples (for plotting)
    #[arg(long)]
    progress_csv: Option<PathBuf>,

    /// Enable Chrome trace event output (viewable in chrome://tracing or Perfetto)
    #[arg(long)]
    chrome_trace: Option<PathBuf>,
}

fn default_cache_dir() -> PathBuf {
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    home.join(".cache/lance-bench/ivf-index")
}

fn parse_distance_type(s: &str) -> Result<DistanceType> {
    match s.to_lowercase().as_str() {
        "l2" | "euclidean" => Ok(DistanceType::L2),
        "cosine" => Ok(DistanceType::Cosine),
        "dot" => Ok(DistanceType::Dot),
        other => anyhow::bail!(
            "Unknown distance type '{}'. Choose from: l2, cosine, dot",
            other,
        ),
    }
}

// ---------------------------------------------------------------------------
// JSON output
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct BenchmarkOutput {
    benchmark_type: String,
    timestamp: u64,
    results: Vec<BenchmarkResult>,
}

#[derive(Serialize)]
struct BenchmarkResult {
    benchmark_name: String,
    num_vectors: usize,
    dimensions: usize,
    num_partitions: usize,
    num_sub_vectors: usize,
    distance_type: String,
    duration_ns: u64,
    peak_rss_bytes: u64,
    delta_rss_bytes: u64,
    values_ns: Vec<u64>,
}

// ---------------------------------------------------------------------------
// Lance dataset creation (random vectors)
// ---------------------------------------------------------------------------

/// Cached metadata written alongside the Lance dataset.
#[derive(Serialize, serde::Deserialize)]
struct DatasetMeta {
    num_vectors: usize,
    dimensions: usize,
}

/// Ensure a Lance dataset with random vectors exists on disk, generating if needed.
async fn ensure_lance_dataset(
    num_vectors: usize,
    dimensions: usize,
    cache_dir: &Path,
    force_recreate: bool,
) -> Result<(PathBuf, DatasetMeta)> {
    let dataset_key = format!("random_{}v_{}d", num_vectors, dimensions);
    let lance_path = cache_dir.join(format!("{}.lance", dataset_key));
    let meta_path = cache_dir.join(format!("{}.meta.json", dataset_key));

    // Check cache
    if !force_recreate && lance_path.exists() && meta_path.exists() {
        let meta: DatasetMeta = serde_json::from_str(&std::fs::read_to_string(&meta_path)?)?;
        println!(
            "  \u{2713} Reusing cached Lance dataset ({} vectors, {} dims)",
            meta.num_vectors, meta.dimensions,
        );
        return Ok((lance_path, meta));
    }

    println!(
        "  \u{2139}\u{fe0f} Generating {} random vectors of dimension {}...",
        num_vectors, dimensions,
    );

    let batch_size: usize = 10_000;
    let num_batches = (num_vectors + batch_size - 1) / batch_size;
    // The last batch may overshoot, so we generate exact num_batches batches of batch_size
    // and the total will be num_batches * batch_size. We'll adjust if needed.
    let total_rows = num_batches * batch_size;

    let reader = gen_batch()
        .col(
            "vector",
            array::rand_vec::<Float32Type>(Dimension::from(dimensions as u32)),
        )
        .into_reader_rows(
            RowCount::from(batch_size as u64),
            BatchCount::from(num_batches as u32),
        );

    if lance_path.exists() {
        std::fs::remove_dir_all(&lance_path)?;
    }

    Dataset::write(
        reader,
        lance_path.to_str().context("Invalid cache path")?,
        None,
    )
    .await?;

    let actual_rows = total_rows.min(num_vectors + batch_size - 1);
    let meta = DatasetMeta {
        num_vectors: actual_rows,
        dimensions,
    };
    std::fs::write(&meta_path, serde_json::to_string(&meta)?)?;

    println!(
        "  \u{2713} Lance dataset written ({} vectors, {} dims, {:.1} MB)",
        actual_rows,
        dimensions,
        (actual_rows * dimensions * 4) as f64 / 1_000_000.0,
    );
    Ok((lance_path, meta))
}

// ---------------------------------------------------------------------------
// Index build progress reporting
// ---------------------------------------------------------------------------

/// A single resource-usage sample captured on a fixed 500ms interval.
struct ProgressSample {
    elapsed_s: f64,
    stage: String,
    event: String,
    progress_value: u64,
    rss_bytes: u64,
    /// Average number of CPUs actively used since the previous sample.
    cpu_active: f64,
    /// Additional disk space used by the Lance dataset since the index build started.
    disk_delta_bytes: u64,
    /// Number of slow polls (exceeding threshold) since the previous sample.
    slow_poll_count: u64,
    /// Mean duration of slow polls (ns) since the previous sample.
    mean_slow_poll_ns: f64,
    /// Number of long scheduling delays since the previous sample.
    long_delay_count: u64,
    /// Mean duration of long scheduling delays (ns) since the previous sample.
    mean_long_delay_ns: f64,
    /// Number of alive tokio tasks at sample time.
    num_alive_tasks: usize,
    /// Number of tasks queued (injection + all worker local queues).
    queued_tasks: usize,
}

/// Current stage state, updated by progress callbacks and read by the polling task.
struct StageState {
    stage: String,
    event: String,
    progress_value: u64,
}

/// Shared state between the polling task and the progress callbacks.
struct ProgressMonitorState {
    start: Instant,
    disk_paths: Vec<PathBuf>,
    disk_baseline: u64,
    prev_elapsed: Mutex<f64>,
    prev_cpu: Mutex<f64>,
    samples: Mutex<Vec<ProgressSample>>,
    metrics_iter: Mutex<Box<dyn Iterator<Item = tokio_metrics::TaskMetrics> + Send>>,
    current_stage: Mutex<StageState>,
    stop: tokio::sync::Notify,
}

impl std::fmt::Debug for ProgressMonitorState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProgressMonitorState")
            .field("start", &self.start)
            .finish()
    }
}

impl ProgressMonitorState {
    fn disk_total(&self) -> u64 {
        self.disk_paths.iter().map(|p| get_dir_size_bytes(p)).sum()
    }

    fn capture(&self) {
        let elapsed_s = self.start.elapsed().as_secs_f64();
        let rss_bytes = get_rss_bytes();
        let cpu_now = get_cpu_time_secs();
        let disk_delta_bytes = self.disk_total().saturating_sub(self.disk_baseline);

        let mut prev_e = self.prev_elapsed.lock().unwrap();
        let mut prev_c = self.prev_cpu.lock().unwrap();

        let dt = elapsed_s - *prev_e;
        let dc = cpu_now - *prev_c;
        let cpu_active = if dt > 0.001 { dc / dt } else { 0.0 };

        *prev_e = elapsed_s;
        *prev_c = cpu_now;

        // Poll tokio task metrics (delta since last sample)
        let task_metrics = self.metrics_iter.lock().unwrap().next();
        let (slow_poll_count, mean_slow_poll_ns, long_delay_count, mean_long_delay_ns) =
            match task_metrics {
                Some(m) => (
                    m.total_slow_poll_count,
                    m.mean_slow_poll_duration().as_nanos() as f64,
                    m.total_long_delay_count,
                    m.mean_long_delay_duration().as_nanos() as f64,
                ),
                None => (0, 0.0, 0, 0.0),
            };

        // Runtime-wide task counts (requires tokio_unstable)
        let rt_metrics = tokio::runtime::Handle::current().metrics();
        let num_alive_tasks = rt_metrics.num_alive_tasks();
        let mut queued_tasks = rt_metrics.global_queue_depth();
        for w in 0..rt_metrics.num_workers() {
            queued_tasks += rt_metrics.worker_local_queue_depth(w);
        }

        // Read current stage state
        let stage_state = self.current_stage.lock().unwrap();
        let stage = stage_state.stage.clone();
        let event = stage_state.event.clone();
        let progress_value = stage_state.progress_value;
        drop(stage_state);

        // Print status line
        let rss_mb = rss_bytes as f64 / 1_000_000.0;
        let disk_delta_mb = disk_delta_bytes as f64 / 1_000_000.0;
        println!(
            "    [{:.1}s] {}/{}: progress={} ({:.0} MB RSS, +{:.0} MB disk, {} slow polls, {} alive tasks, {} queued)",
            elapsed_s, stage, event, progress_value, rss_mb, disk_delta_mb,
            slow_poll_count, num_alive_tasks, queued_tasks,
        );

        self.samples.lock().unwrap().push(ProgressSample {
            elapsed_s,
            stage,
            event,
            progress_value,
            rss_bytes,
            cpu_active,
            disk_delta_bytes,
            slow_poll_count,
            mean_slow_poll_ns,
            long_delay_count,
            mean_long_delay_ns,
            num_alive_tasks,
            queued_tasks,
        });
    }
}

/// Manages periodic sampling and exposes progress callbacks that update stage state.
///
/// A background task captures resource metrics every 500ms. The `IndexBuildProgress`
/// callbacks only update the current stage/event/progress_value — they don't trigger
/// captures themselves.
#[derive(Debug)]
struct ConsoleProgress {
    state: Arc<ProgressMonitorState>,
}

impl ConsoleProgress {
    fn new(lance_path: PathBuf, task_monitor: &TaskMonitor) -> Self {
        let cpu_now = get_cpu_time_secs();
        let disk_paths = vec![lance_path, PathBuf::from("/tmp")];
        let disk_baseline = disk_paths.iter().map(|p| get_dir_size_bytes(p)).sum();
        Self {
            state: Arc::new(ProgressMonitorState {
                start: Instant::now(),
                disk_paths,
                disk_baseline,
                prev_elapsed: Mutex::new(0.0),
                prev_cpu: Mutex::new(cpu_now),
                samples: Mutex::new(Vec::new()),
                metrics_iter: Mutex::new(Box::new(task_monitor.intervals())),
                current_stage: Mutex::new(StageState {
                    stage: String::new(),
                    event: String::new(),
                    progress_value: 0,
                }),
                stop: tokio::sync::Notify::new(),
            }),
        }
    }

    /// Spawn the background polling task. Returns its join handle.
    fn start_polling(&self) -> tokio::task::JoinHandle<()> {
        let state = self.state.clone();
        tokio::task::spawn_blocking(move || {
            let rt = tokio::runtime::Handle::current();
            loop {
                // Wait 500ms or until signalled to stop
                let should_stop = rt.block_on(async {
                    tokio::select! {
                        _ = state.stop.notified() => true,
                        _ = tokio::time::sleep(std::time::Duration::from_millis(500)) => false,
                    }
                });
                if should_stop {
                    break;
                }
                state.capture();
            }
            // One final capture
            state.capture();
        })
    }

    /// Signal the polling task to stop.
    fn stop_polling(&self) {
        self.state.stop.notify_one();
    }

    fn into_samples(self) -> Vec<ProgressSample> {
        Arc::try_unwrap(self.state)
            .expect("progress Arc should have no other owners")
            .samples
            .into_inner()
            .unwrap()
    }
}

#[async_trait::async_trait]
impl IndexBuildProgress for ConsoleProgress {
    async fn stage_start(&self, stage: &str, total: Option<u64>, _unit: &str) -> lance::Result<()> {
        let mut s = self.state.current_stage.lock().unwrap();
        s.stage = stage.to_string();
        s.event = "start".to_string();
        s.progress_value = total.unwrap_or(0);
        Ok(())
    }

    #[tracing::instrument(skip(self))]
    async fn stage_progress(&self, stage: &str, completed: u64) -> lance::Result<()> {
        let mut s = self.state.current_stage.lock().unwrap();
        s.stage = stage.to_string();
        s.event = "progress".to_string();
        s.progress_value = completed;
        Ok(())
    }

    async fn stage_complete(&self, stage: &str) -> lance::Result<()> {
        let mut s = self.state.current_stage.lock().unwrap();
        s.stage = stage.to_string();
        s.event = "complete".to_string();
        s.progress_value = 0;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Benchmark execution
// ---------------------------------------------------------------------------

async fn run_benchmark(
    num_vectors: usize,
    dimensions: usize,
    num_partitions: usize,
    distance_type: DistanceType,
    cache_dir: &Path,
    force_recreate: bool,
) -> Result<(BenchmarkResult, Vec<ProgressSample>)> {
    let num_sub_vectors = dimensions / 16;
    let dt_str = format!("{}", distance_type).to_lowercase();
    let bench_name = format!(
        "train_ivf_index/vectors={}/dim={}/partitions={}/distance={}",
        num_vectors, dimensions, num_partitions, dt_str,
    );

    println!("\n{}", "=".repeat(72));
    println!("  Benchmark: {}", bench_name);
    println!(
        "  num_sub_vectors={} (dim/{} = {}/{})",
        num_sub_vectors, 16, dimensions, 16,
    );
    println!("{}", "=".repeat(72));

    // Ensure dataset exists
    let (lance_path, _meta) =
        ensure_lance_dataset(num_vectors, dimensions, cache_dir, force_recreate).await?;

    // Open dataset fresh for indexing
    let mut dataset = Dataset::open(lance_path.to_str().unwrap()).await?;

    // Configure IVF/PQ index params
    let params = lance::index::vector::VectorIndexParams::ivf_pq(
        num_partitions,
        8, // num_bits (always 8 for PQ)
        num_sub_vectors,
        distance_type,
        50, // max kmeans iterations
    );

    let task_monitor = TaskMonitor::new();
    let progress = Arc::new(ConsoleProgress::new(lance_path.clone(), &task_monitor));
    let progress_trait: Arc<dyn IndexBuildProgress> = progress.clone();

    // Start periodic sampling (500ms intervals)
    let poll_handle = progress.start_polling();

    // Measure
    let mut monitor = PeakRssMonitor::new();
    monitor.start();

    let start = Instant::now();
    task_monitor
        .instrument(async {
            dataset
                .create_index_builder(&["vector"], IndexType::Vector, &params)
                .replace(true)
                .progress(progress_trait)
                .await
        })
        .await?;
    let duration_ns = start.elapsed().as_nanos() as u64;

    // Stop polling and wait for final capture
    progress.stop_polling();
    poll_handle.await?;

    let (peak_rss, delta_rss) = monitor.stop();

    let duration_s = duration_ns as f64 / 1_000_000_000.0;
    println!(
        "  \u{2713} Index created in {:.2}s (distance={})",
        duration_s, dt_str,
    );
    println!(
        "  \u{2713} Peak RSS: {:.0} MB, delta RSS: {:.0} MB",
        peak_rss as f64 / 1_000_000.0,
        delta_rss as f64 / 1_000_000.0,
    );

    // Extract progress samples — unwrap the Arc (we hold the only remaining ref)
    let samples = Arc::try_unwrap(progress)
        .expect("progress Arc should have no other owners")
        .into_samples();

    Ok((
        BenchmarkResult {
            benchmark_name: bench_name,
            num_vectors,
            dimensions,
            num_partitions,
            num_sub_vectors,
            distance_type: dt_str,
            duration_ns,
            peak_rss_bytes: peak_rss,
            delta_rss_bytes: delta_rss,
            values_ns: vec![duration_ns],
        },
        samples,
    ))
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Validate dimensions is divisible by 16
    if args.dimensions % 16 != 0 {
        anyhow::bail!(
            "Dimensions ({}) must be divisible by 16 for PQ sub-vector computation",
            args.dimensions,
        );
    }

    // Set up tracing: either Chrome trace output or plain env_logger
    let _chrome_guard = if let Some(trace_path) = &args.chrome_trace {
        if let Some(parent) = trace_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let (chrome_layer, guard) = tracing_chrome::ChromeLayerBuilder::new()
            .file(trace_path.clone())
            .include_args(true)
            .build();
        use tracing_subscriber::layer::SubscriberExt;
        use tracing_subscriber::util::SubscriberInitExt;
        tracing_subscriber::registry()
            .with(chrome_layer)
            .with(tracing_subscriber::EnvFilter::from_default_env())
            .init();
        println!(
            "\u{2139}\u{fe0f} Chrome trace output enabled: {}",
            trace_path.display(),
        );
        Some(guard)
    } else {
        env_logger::init();
        None
    };

    // Resolve distance type variants
    let dt_values: Vec<DistanceType> = match args.distance_type.as_str() {
        "all" => vec![DistanceType::L2, DistanceType::Cosine, DistanceType::Dot],
        other => vec![parse_distance_type(other)?],
    };

    // Resolve num_partitions: default to min(num_vectors / 4096, sqrt(num_vectors))
    let num_partitions = args.num_partitions.unwrap_or_else(|| {
        let by_ratio = args.num_vectors / 4096;
        let by_sqrt = (args.num_vectors as f64).sqrt() as usize;
        by_ratio.min(by_sqrt).max(1)
    });

    // Ensure cache directory exists
    std::fs::create_dir_all(&args.cache_dir)?;

    println!("\u{2139}\u{fe0f} IVF/PQ Index Training Benchmark");
    println!("  Vectors: {}", args.num_vectors);
    println!("  Dimensions: {}", args.dimensions);
    println!("  Partitions: {}", num_partitions);
    println!("  Sub-vectors: {}", args.dimensions / 16);
    println!("  Distance types: {:?}", dt_values);
    println!("  Cache dir: {}", args.cache_dir.display());

    let mut results = Vec::new();
    let mut all_samples: Vec<(String, ProgressSample)> = Vec::new();

    for &dt in &dt_values {
        let (result, samples) = run_benchmark(
            args.num_vectors,
            args.dimensions,
            num_partitions,
            dt,
            &args.cache_dir,
            args.force_recreate,
        )
        .await?;
        let bench_name = result.benchmark_name.clone();
        results.push(result);
        all_samples.extend(samples.into_iter().map(|s| (bench_name.clone(), s)));
    }

    // Write JSON output
    let output = BenchmarkOutput {
        benchmark_type: "train_ivf_index".to_string(),
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs(),
        results,
    };

    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&args.output, serde_json::to_string_pretty(&output)?)?;

    println!("\n\u{2713} Results written to {}", args.output.display());
    println!("  {} benchmark result(s) total", output.results.len());

    // Write progress CSV if requested
    if let Some(csv_path) = &args.progress_csv {
        if let Some(parent) = csv_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut file = std::fs::File::create(csv_path)?;
        writeln!(
            file,
            "benchmark_name,elapsed_s,stage,event,progress_value,rss_bytes,rss_mb,cpu_active,disk_delta_bytes,disk_delta_mb,slow_poll_count,mean_slow_poll_ns,long_delay_count,mean_long_delay_ns,num_alive_tasks,queued_tasks"
        )?;
        for (name, s) in &all_samples {
            writeln!(
                file,
                "{},{:.3},{},{},{},{},{:.1},{:.2},{},{:.1},{},{:.0},{},{:.0},{},{}",
                name,
                s.elapsed_s,
                s.stage,
                s.event,
                s.progress_value,
                s.rss_bytes,
                s.rss_bytes as f64 / 1_000_000.0,
                s.cpu_active,
                s.disk_delta_bytes,
                s.disk_delta_bytes as f64 / 1_000_000.0,
                s.slow_poll_count,
                s.mean_slow_poll_ns,
                s.long_delay_count,
                s.mean_long_delay_ns,
                s.num_alive_tasks,
                s.queued_tasks,
            )?;
        }
        println!(
            "\u{2713} Progress CSV written to {} ({} samples)",
            csv_path.display(),
            all_samples.len(),
        );
    }

    Ok(())
}
