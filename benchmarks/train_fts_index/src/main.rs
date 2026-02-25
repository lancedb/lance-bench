//! FTS Index Training Benchmark
//!
//! Benchmarks how long it takes (and how much RAM it uses) to train a
//! full-text search (inverted) index on Lance datasets.
//!
//! Uses two HuggingFace datasets representing different text-per-row profiles:
//! - FineWeb (large documents from Common Crawl)
//! - HH-RLHF (short dialogue prompts)

use anyhow::{Context, Result};
use arrow_array::{Array, LargeStringArray, RecordBatch, RecordBatchIterator, StringArray};
use arrow_schema::{ArrowError, DataType, Field, Schema};
use clap::Parser;
use futures::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use lance::Dataset;
use lance_index::progress::IndexBuildProgress;
use lance_index::scalar::InvertedIndexParams;
use lance_index::{DatasetIndexExt, IndexType};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ProjectionMask;
use serde::Serialize;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tokio::io::AsyncWriteExt;
use tokio_metrics::TaskMonitor;

mod memory;
use memory::{get_cpu_time_secs, get_rss_bytes, PeakRssMonitor};

/// Recursively compute the total size of all files under a directory.
fn get_dir_size_bytes(path: &Path) -> u64 {
    walkdir(path).unwrap()
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
// Dataset definitions
// ---------------------------------------------------------------------------

struct DatasetConfig {
    key: &'static str,
    hf_id: &'static str,
    hf_config: &'static str,
    column: &'static str,
    default_rows: usize,
    description: &'static str,
}

const DATASETS: &[DatasetConfig] = &[
    DatasetConfig {
        key: "fineweb",
        hf_id: "HuggingFaceFW/fineweb",
        hf_config: "sample-10BT",
        column: "text",
        default_rows: 10_000_000,
        description: "Large text per row: FineWeb web crawl documents",
    },
    DatasetConfig {
        key: "hh-rlhf",
        hf_id: "Anthropic/hh-rlhf",
        hf_config: "default",
        column: "chosen",
        default_rows: 1_610_000,
        description: "Small text per row: HH-RLHF dialogue prompts",
    },
];

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser, Debug)]
#[command(name = "train-fts-index-benchmark")]
#[command(about = "Benchmark FTS (inverted) index training on Lance datasets")]
struct Args {
    /// Which dataset(s) to benchmark: fineweb, hh-rlhf, or all
    #[arg(long, default_value = "all")]
    dataset: String,

    /// Override the default row count for each dataset
    #[arg(long)]
    num_rows: Option<usize>,

    /// JSON output path
    #[arg(long, default_value = "fts-index-results.json")]
    output: PathBuf,

    /// Cache directory for HF data and Lance datasets
    #[arg(long, default_value_os_t = default_cache_dir())]
    cache_dir: PathBuf,

    /// Benchmark with_position: true, false, or both
    #[arg(long, default_value = "both")]
    with_position: String,

    /// Force re-download and re-creation of Lance datasets
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
    home.join(".cache/lance-bench/fts-index")
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
    dataset_name: String,
    dataset_description: String,
    num_rows: usize,
    total_text_bytes: u64,
    duration_ns: u64,
    peak_rss_bytes: u64,
    delta_rss_bytes: u64,
    with_position: bool,
    values_ns: Vec<u64>,
}

// ---------------------------------------------------------------------------
// HuggingFace download helpers
// ---------------------------------------------------------------------------

/// Fetch parquet file URLs from the HuggingFace Hub API.
async fn get_parquet_urls(
    client: &reqwest::Client,
    dataset_id: &str,
    config: &str,
) -> Result<Vec<String>> {
    let url = format!(
        "https://huggingface.co/api/datasets/{}/parquet/{}/train",
        dataset_id, config,
    );
    println!("    Fetching parquet file list...");
    let response = client
        .get(&url)
        .send()
        .await?
        .error_for_status()
        .context("Failed to fetch parquet URLs from HuggingFace")?;
    let urls: Vec<String> = response.json().await?;
    println!("    Found {} parquet file(s)", urls.len());
    Ok(urls)
}

/// Stream-download a file from URL to a local path, showing a progress bar.
async fn download_file(client: &reqwest::Client, url: &str, dest: &Path) -> Result<()> {
    println!("    Downloading to {}...", dest.display());
    let response = client.get(url).send().await?.error_for_status()?;
    let total = response.content_length().unwrap_or(0);

    let pb = ProgressBar::new(total);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("    [{bar:40}] {bytes}/{total_bytes} ({bytes_per_sec})")
            .unwrap(),
    );

    let mut file = tokio::fs::File::create(dest).await?;
    let mut stream = response.bytes_stream();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        file.write_all(&chunk).await?;
        pb.inc(chunk.len() as u64);
    }

    file.flush().await?;
    pb.finish_and_clear();
    Ok(())
}

/// Count total text bytes in a string column (handles Utf8 and LargeUtf8).
fn count_text_bytes(col: &dyn Array) -> u64 {
    match col.data_type() {
        DataType::Utf8 => {
            let arr = col.as_any().downcast_ref::<StringArray>().unwrap();
            (0..arr.len())
                .filter(|&i| !arr.is_null(i))
                .map(|i| arr.value(i).len() as u64)
                .sum()
        }
        DataType::LargeUtf8 => {
            let arr = col.as_any().downcast_ref::<LargeStringArray>().unwrap();
            (0..arr.len())
                .filter(|&i| !arr.is_null(i))
                .map(|i| arr.value(i).len() as u64)
                .sum()
        }
        _ => 0,
    }
}

/// Streams record batches from multiple parquet shards, projecting to a single
/// text column, renaming it to "text", and enforcing a row limit.
struct ShardedParquetReader {
    shard_paths: Vec<PathBuf>,
    column_name: String,
    max_rows: usize,
    out_schema: Arc<Schema>,
    shard_idx: usize,
    reader: Option<parquet::arrow::arrow_reader::ParquetRecordBatchReader>,
    rows_emitted: usize,
    row_counter: Arc<AtomicU64>,
    byte_counter: Arc<AtomicU64>,
}

impl ShardedParquetReader {
    fn new(
        shard_paths: Vec<PathBuf>,
        column_name: String,
        max_rows: usize,
        out_schema: Arc<Schema>,
        row_counter: Arc<AtomicU64>,
        byte_counter: Arc<AtomicU64>,
    ) -> Self {
        Self {
            shard_paths,
            column_name,
            max_rows,
            out_schema,
            shard_idx: 0,
            reader: None,
            rows_emitted: 0,
            row_counter,
            byte_counter,
        }
    }

    fn open_shard(
        &self,
        path: &Path,
    ) -> std::result::Result<parquet::arrow::arrow_reader::ParquetRecordBatchReader, ArrowError>
    {
        let file = std::fs::File::open(path).map_err(|e| ArrowError::ExternalError(Box::new(e)))?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let parquet_schema = builder.parquet_schema();
        let col_idx = parquet_schema
            .columns()
            .iter()
            .position(|c| c.name() == self.column_name)
            .ok_or_else(|| {
                ArrowError::SchemaError(format!(
                    "Column '{}' not found in parquet file",
                    self.column_name
                ))
            })?;
        let projection = ProjectionMask::roots(parquet_schema, [col_idx]);
        Ok(builder.with_projection(projection).build()?)
    }
}

impl Iterator for ShardedParquetReader {
    type Item = std::result::Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.rows_emitted >= self.max_rows {
            return None;
        }

        loop {
            if let Some(reader) = &mut self.reader {
                match reader.next() {
                    Some(Ok(batch)) => {
                        let remaining = self.max_rows - self.rows_emitted;
                        let batch = if batch.num_rows() > remaining {
                            batch.slice(0, remaining)
                        } else {
                            batch
                        };
                        let col = batch.column(0).clone();
                        let bytes = count_text_bytes(col.as_ref());
                        let new_batch = RecordBatch::try_new(self.out_schema.clone(), vec![col]);
                        match new_batch {
                            Ok(b) => {
                                let n = b.num_rows() as u64;
                                self.rows_emitted += n as usize;
                                self.row_counter.fetch_add(n, Ordering::Relaxed);
                                self.byte_counter.fetch_add(bytes, Ordering::Relaxed);
                                return Some(Ok(b));
                            }
                            Err(e) => return Some(Err(e)),
                        }
                    }
                    Some(Err(e)) => return Some(Err(e)),
                    None => self.reader = None,
                }
            }

            if self.shard_idx >= self.shard_paths.len() {
                return None;
            }

            let path = self.shard_paths[self.shard_idx].clone();
            self.shard_idx += 1;
            match self.open_shard(&path) {
                Ok(r) => self.reader = Some(r),
                Err(e) => return Some(Err(e)),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Lance dataset creation
// ---------------------------------------------------------------------------

/// Cached metadata written alongside the Lance dataset.
#[derive(Serialize, serde::Deserialize)]
struct DatasetMeta {
    num_rows: usize,
    total_text_bytes: u64,
}

/// Ensure a Lance dataset exists on disk, downloading from HuggingFace if needed.
///
/// Streams parquet data directly into the Lance dataset to avoid loading
/// all rows into memory at once.
async fn ensure_lance_dataset(
    cfg: &DatasetConfig,
    num_rows: usize,
    cache_dir: &Path,
    force_recreate: bool,
    client: &reqwest::Client,
) -> Result<(PathBuf, DatasetMeta)> {
    let lance_path = cache_dir.join(format!("{}.lance", cfg.key));
    let meta_path = cache_dir.join(format!("{}.meta.json", cfg.key));

    // Check cache
    if !force_recreate && lance_path.exists() && meta_path.exists() {
        let meta: DatasetMeta = serde_json::from_str(&std::fs::read_to_string(&meta_path)?)?;
        println!(
            "  ✓ Reusing cached Lance dataset ({} rows, {} bytes)",
            meta.num_rows, meta.total_text_bytes,
        );
        return Ok((lance_path, meta));
    }

    // Download parquet shards from HuggingFace until we have enough rows
    let parquet_dir = cache_dir.join("parquet").join(cfg.key);
    std::fs::create_dir_all(&parquet_dir)?;

    println!("  ℹ️ Downloading {} from HuggingFace...", cfg.hf_id);
    let urls = get_parquet_urls(client, cfg.hf_id, cfg.hf_config).await?;
    if urls.is_empty() {
        anyhow::bail!("No parquet files found for {}", cfg.hf_id);
    }

    let mut shard_paths = Vec::new();
    let mut available_rows: usize = 0;
    for (i, url) in urls.iter().enumerate() {
        let shard_path = parquet_dir.join(format!("shard_{i:04}.parquet"));
        if !shard_path.exists() || force_recreate {
            println!("  ℹ️ Downloading shard {}/{} ...", i + 1, urls.len(),);
            download_file(client, url, &shard_path).await?;
        }
        // Read row count from parquet metadata (no data read)
        let f = std::fs::File::open(&shard_path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(f)?;
        let shard_rows = builder.metadata().file_metadata().num_rows() as usize;
        available_rows += shard_rows;
        shard_paths.push(shard_path);
        println!(
            "    Shard {}: {} rows (cumulative: {})",
            i, shard_rows, available_rows,
        );
        if available_rows >= num_rows {
            break;
        }
    }
    println!(
        "  ✓ {} shard(s) ready, {} rows available",
        shard_paths.len(),
        available_rows,
    );

    // Detect column type from first shard
    let first_file = std::fs::File::open(&shard_paths[0])?;
    let first_builder = ParquetRecordBatchReaderBuilder::try_new(first_file)?;
    let arrow_schema = first_builder.schema();
    let source_field = arrow_schema
        .field_with_name(cfg.column)
        .context("Column not found in parquet schema")?;
    let text_type = source_field.data_type().clone();
    let out_schema = Arc::new(Schema::new(vec![Field::new("text", text_type, false)]));

    // Stream parquet batches directly into Lance
    println!(
        "  ℹ️ Streaming up to {} rows into Lance at {}...",
        num_rows,
        lance_path.display(),
    );
    let row_counter = Arc::new(AtomicU64::new(0));
    let byte_counter = Arc::new(AtomicU64::new(0));

    let streamer = ShardedParquetReader::new(
        shard_paths,
        cfg.column.to_string(),
        num_rows,
        out_schema.clone(),
        row_counter.clone(),
        byte_counter.clone(),
    );
    let reader = RecordBatchIterator::new(streamer, out_schema);

    if lance_path.exists() {
        std::fs::remove_dir_all(&lance_path)?;
    }
    Dataset::write(reader, lance_path.to_str().unwrap(), None).await?;

    let final_rows = row_counter.load(Ordering::Relaxed) as usize;
    let final_bytes = byte_counter.load(Ordering::Relaxed);

    let meta = DatasetMeta {
        num_rows: final_rows,
        total_text_bytes: final_bytes,
    };
    std::fs::write(&meta_path, serde_json::to_string(&meta)?)?;

    println!(
        "  ✓ Lance dataset written ({} rows, {:.1} MB text)",
        final_rows,
        final_bytes as f64 / 1_000_000.0,
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
    cfg: &DatasetConfig,
    num_rows: usize,
    cache_dir: &Path,
    force_recreate: bool,
    with_position: bool,
    client: &reqwest::Client,
) -> Result<(BenchmarkResult, Vec<ProgressSample>)> {
    let dataset_name = cfg.key.replace('-', "_");
    let bench_name = format!(
        "train_fts_index/{}/with_position={}",
        dataset_name, with_position,
    );

    println!("\n{}", "=".repeat(60));
    println!("  Benchmark: {}", bench_name);
    println!("{}", "=".repeat(60));

    // Ensure dataset exists
    let (lance_path, meta) =
        ensure_lance_dataset(cfg, num_rows, cache_dir, force_recreate, client).await?;

    // Open dataset fresh for indexing
    let mut dataset = Dataset::open(lance_path.to_str().unwrap()).await?;

    // Configure index params
    let params = InvertedIndexParams::default().with_position(with_position);
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
                .create_index_builder(&["text"], IndexType::Inverted, &params)
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
        "  ✓ Index created in {:.2}s (with_position={})",
        duration_s, with_position,
    );
    println!(
        "  ✓ Peak RSS: {:.0} MB, delta RSS: {:.0} MB",
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
            dataset_name,
            dataset_description: cfg.description.to_string(),
            num_rows: meta.num_rows,
            total_text_bytes: meta.total_text_bytes,
            duration_ns,
            peak_rss_bytes: peak_rss,
            delta_rss_bytes: delta_rss,
            with_position,
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
        println!("ℹ️ Chrome trace output enabled: {}", trace_path.display(),);
        Some(guard)
    } else {
        env_logger::init();
        None
    };

    // Resolve which datasets to run
    let dataset_keys: Vec<&str> = if args.dataset == "all" {
        DATASETS.iter().map(|d| d.key).collect()
    } else {
        vec![args.dataset.as_str()]
    };

    // Validate dataset selection
    for &key in &dataset_keys {
        if !DATASETS.iter().any(|d| d.key == key) {
            anyhow::bail!(
                "Unknown dataset '{}'. Choose from: fineweb, hh-rlhf, all",
                key,
            );
        }
    }

    // Resolve with_position variants
    let wp_values: Vec<bool> = match args.with_position.as_str() {
        "both" => vec![true, false],
        "true" => vec![true],
        "false" => vec![false],
        other => anyhow::bail!(
            "Invalid --with-position value: '{}'. Use true, false, or both",
            other,
        ),
    };

    // Ensure cache directory exists
    std::fs::create_dir_all(&args.cache_dir)?;

    println!("ℹ️ FTS Index Training Benchmark");
    println!("  Datasets: {:?}", dataset_keys);
    println!("  with_position: {:?}", wp_values);
    println!("  Cache dir: {}", args.cache_dir.display());

    let client = reqwest::Client::builder()
        .user_agent("lance-bench/0.1.0")
        .build()?;

    let mut results = Vec::new();
    let mut all_samples: Vec<(String, ProgressSample)> = Vec::new();

    for &key in &dataset_keys {
        let cfg = DATASETS.iter().find(|d| d.key == key).unwrap();
        let num_rows = args.num_rows.unwrap_or(cfg.default_rows);

        for &wp in &wp_values {
            let (result, samples) = run_benchmark(
                cfg,
                num_rows,
                &args.cache_dir,
                args.force_recreate,
                wp,
                &client,
            )
            .await?;
            let bench_name = result.benchmark_name.clone();
            results.push(result);
            all_samples.extend(samples.into_iter().map(|s| (bench_name.clone(), s)));
        }
    }

    // Write JSON output
    let output = BenchmarkOutput {
        benchmark_type: "train_fts_index".to_string(),
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs(),
        results,
    };

    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&args.output, serde_json::to_string_pretty(&output)?)?;

    println!("\n✓ Results written to {}", args.output.display());
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
            "✓ Progress CSV written to {} ({} samples)",
            csv_path.display(),
            all_samples.len(),
        );
    }

    Ok(())
}
