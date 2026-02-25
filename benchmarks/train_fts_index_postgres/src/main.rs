//! FTS Index Training Benchmark — PostgreSQL (GIN + tsvector)
//!
//! Benchmarks how long it takes (and how much RAM it uses) to build a
//! full-text search GIN index on an embedded PostgreSQL instance, using
//! the same HuggingFace datasets as the Lance FTS benchmark.
//!
//! Uses two HuggingFace datasets representing different text-per-row profiles:
//! - FineWeb (large documents from Common Crawl)
//! - HH-RLHF (short dialogue prompts)

use anyhow::{Context, Result};
use arrow_array::{Array, LargeStringArray, RecordBatch, StringArray};
use arrow_schema::DataType;
use clap::Parser;
use futures::SinkExt;
use indicatif::{ProgressBar, ProgressStyle};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ProjectionMask;
use postgresql_embedded::PostgreSQL;
use serde::Serialize;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tokio::io::AsyncWriteExt;
use tokio_postgres::NoTls;

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
        default_rows: 50_000_000,
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
#[command(name = "train-fts-index-postgres-benchmark")]
#[command(about = "Benchmark FTS (GIN) index training on PostgreSQL with embedded server")]
struct Args {
    /// Which dataset(s) to benchmark: fineweb, hh-rlhf, or all
    #[arg(long, default_value = "all")]
    dataset: String,

    /// Override the default row count for each dataset
    #[arg(long)]
    num_rows: Option<usize>,

    /// JSON output path
    #[arg(long, default_value = "fts-index-postgres-results.json")]
    output: PathBuf,

    /// Cache directory for HF data
    #[arg(long, default_value_os_t = default_cache_dir())]
    cache_dir: PathBuf,

    /// Force re-download of parquet files
    #[arg(long)]
    force_recreate: bool,

    /// Path for CSV output of per-poll RSS/CPU samples (for plotting)
    #[arg(long)]
    progress_csv: Option<PathBuf>,
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

    use futures::StreamExt;
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

/// Extract string values from an Arrow array column (handles Utf8 and LargeUtf8).
fn extract_text_values(col: &dyn Array) -> Vec<Option<String>> {
    match col.data_type() {
        DataType::Utf8 => {
            let arr = col.as_any().downcast_ref::<StringArray>().unwrap();
            (0..arr.len())
                .map(|i| {
                    if arr.is_null(i) {
                        None
                    } else {
                        Some(arr.value(i).to_string())
                    }
                })
                .collect()
        }
        DataType::LargeUtf8 => {
            let arr = col.as_any().downcast_ref::<LargeStringArray>().unwrap();
            (0..arr.len())
                .map(|i| {
                    if arr.is_null(i) {
                        None
                    } else {
                        Some(arr.value(i).to_string())
                    }
                })
                .collect()
        }
        _ => vec![],
    }
}

// ---------------------------------------------------------------------------
// Parquet shard download + reading
// ---------------------------------------------------------------------------

/// Download enough parquet shards to cover num_rows, returning shard paths and total text bytes.
async fn download_parquet_shards(
    cfg: &DatasetConfig,
    num_rows: usize,
    cache_dir: &Path,
    force_recreate: bool,
    client: &reqwest::Client,
) -> Result<(Vec<PathBuf>, u64)> {
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
            println!("  ℹ️ Downloading shard {}/{} ...", i + 1, urls.len());
            download_file(client, url, &shard_path).await?;
        }
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

    // Count total text bytes across all shards (up to num_rows)
    let mut total_text_bytes = 0u64;
    let mut rows_counted = 0usize;
    'outer: for path in &shard_paths {
        let file = std::fs::File::open(path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let parquet_schema = builder.parquet_schema().clone();
        let col_idx = parquet_schema
            .columns()
            .iter()
            .position(|c| c.name() == cfg.column)
            .context("Column not found in parquet")?;
        let projection = ProjectionMask::roots(&parquet_schema, [col_idx]);
        let reader = builder.with_projection(projection).build()?;
        for batch in reader {
            let batch = batch?;
            let remaining = num_rows - rows_counted;
            let batch = if batch.num_rows() > remaining {
                batch.slice(0, remaining)
            } else {
                batch
            };
            total_text_bytes += count_text_bytes(batch.column(0).as_ref());
            rows_counted += batch.num_rows();
            if rows_counted >= num_rows {
                break 'outer;
            }
        }
    }

    Ok((shard_paths, total_text_bytes))
}

/// Read batches from parquet shards up to max_rows, yielding (batch_column, batch_row_count).
struct ShardedParquetReader {
    shard_paths: Vec<PathBuf>,
    column_name: String,
    max_rows: usize,
    shard_idx: usize,
    reader: Option<parquet::arrow::arrow_reader::ParquetRecordBatchReader>,
    rows_emitted: usize,
}

impl ShardedParquetReader {
    fn new(shard_paths: Vec<PathBuf>, column_name: String, max_rows: usize) -> Self {
        Self {
            shard_paths,
            column_name,
            max_rows,
            shard_idx: 0,
            reader: None,
            rows_emitted: 0,
        }
    }

    fn open_shard(
        &self,
        path: &Path,
    ) -> std::result::Result<parquet::arrow::arrow_reader::ParquetRecordBatchReader, arrow_schema::ArrowError>
    {
        let file =
            std::fs::File::open(path).map_err(|e| arrow_schema::ArrowError::ExternalError(Box::new(e)))?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let parquet_schema = builder.parquet_schema();
        let col_idx = parquet_schema
            .columns()
            .iter()
            .position(|c| c.name() == self.column_name)
            .ok_or_else(|| {
                arrow_schema::ArrowError::SchemaError(format!(
                    "Column '{}' not found in parquet file",
                    self.column_name
                ))
            })?;
        let projection = ProjectionMask::roots(parquet_schema, [col_idx]);
        Ok(builder.with_projection(projection).build()?)
    }
}

impl Iterator for ShardedParquetReader {
    type Item = std::result::Result<RecordBatch, arrow_schema::ArrowError>;

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
                        self.rows_emitted += batch.num_rows();
                        return Some(Ok(batch));
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
// Polling progress monitor
// ---------------------------------------------------------------------------

struct ProgressSample {
    elapsed_s: f64,
    rss_bytes: u64,
    cpu_active: f64,
    disk_delta_bytes: u64,
}

struct PollingMonitor {
    samples: Arc<Mutex<Vec<ProgressSample>>>,
    stop: Arc<tokio::sync::Notify>,
}

impl PollingMonitor {
    fn new() -> Self {
        Self {
            samples: Arc::new(Mutex::new(Vec::new())),
            stop: Arc::new(tokio::sync::Notify::new()),
        }
    }

    fn start(&self, disk_paths: Vec<PathBuf>, start_time: Instant) -> tokio::task::JoinHandle<()> {
        let samples = self.samples.clone();
        let stop = self.stop.clone();

        let disk_baseline: u64 = disk_paths.iter().map(|p| get_dir_size_bytes(p)).sum();
        let cpu_baseline = get_cpu_time_secs();

        tokio::spawn(async move {
            let mut prev_elapsed = 0.0f64;
            let mut prev_cpu = cpu_baseline;

            loop {
                tokio::select! {
                    _ = stop.notified() => break,
                    _ = tokio::time::sleep(std::time::Duration::from_secs(1)) => {}
                }

                let elapsed_s = start_time.elapsed().as_secs_f64();
                let rss_bytes = get_rss_bytes();
                let cpu_now = get_cpu_time_secs();
                let disk_total: u64 = disk_paths.iter().map(|p| get_dir_size_bytes(p)).sum();
                let disk_delta_bytes = disk_total.saturating_sub(disk_baseline);

                let dt = elapsed_s - prev_elapsed;
                let dc = cpu_now - prev_cpu;
                let cpu_active = if dt > 0.001 { dc / dt } else { 0.0 };

                prev_elapsed = elapsed_s;
                prev_cpu = cpu_now;

                println!(
                    "    [{:.1}s] RSS: {:.0} MB, CPU active: {:.1}, disk delta: +{:.0} MB",
                    elapsed_s,
                    rss_bytes as f64 / 1_000_000.0,
                    cpu_active,
                    disk_delta_bytes as f64 / 1_000_000.0,
                );

                samples.lock().unwrap().push(ProgressSample {
                    elapsed_s,
                    rss_bytes,
                    cpu_active,
                    disk_delta_bytes,
                });
            }
        })
    }

    fn stop_and_take(self) -> Vec<ProgressSample> {
        self.stop.notify_one();
        self.samples.lock().unwrap().drain(..).collect()
    }
}

// ---------------------------------------------------------------------------
// COPY protocol helper
// ---------------------------------------------------------------------------

/// Escape a text value for PostgreSQL COPY TEXT format.
///
/// In COPY TEXT mode:
///   backslash  -> \\
///   newline    -> \n
///   carriage return -> \r
///   tab        -> \t
fn escape_copy_text(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            _ => out.push(ch),
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Benchmark execution
// ---------------------------------------------------------------------------

async fn run_benchmark(
    cfg: &DatasetConfig,
    num_rows: usize,
    cache_dir: &Path,
    force_recreate: bool,
    client: &reqwest::Client,
) -> Result<(BenchmarkResult, Vec<ProgressSample>)> {
    let dataset_name = cfg.key.replace('-', "_");
    let bench_name = format!("train_fts_index_postgres/{}", dataset_name);

    println!("\n{}", "=".repeat(60));
    println!("  Benchmark: {}", bench_name);
    println!("{}", "=".repeat(60));

    // Download parquet shards
    let (shard_paths, total_text_bytes) =
        download_parquet_shards(cfg, num_rows, cache_dir, force_recreate, client).await?;

    // Start embedded PostgreSQL
    println!("  ℹ️ Starting embedded PostgreSQL...");
    let mut postgresql = PostgreSQL::default();
    postgresql.setup().await.context("Failed to setup PostgreSQL")?;
    postgresql.start().await.context("Failed to start PostgreSQL")?;

    let settings = postgresql.settings();
    let pg_data_dir = settings.data_dir.clone();
    println!(
        "  ✓ PostgreSQL started on {}:{} (data dir: {})",
        settings.host, settings.port, pg_data_dir.display(),
    );

    // Create database
    let db_name = "bench";
    postgresql
        .create_database(db_name)
        .await
        .context("Failed to create database")?;

    // Connect with tokio-postgres
    let conn_str = format!(
        "host={} port={} user={} password={} dbname={}",
        settings.host, settings.port, settings.username, settings.password, db_name,
    );
    let (pg_client, connection) = tokio_postgres::connect(&conn_str, NoTls)
        .await
        .context("Failed to connect to PostgreSQL")?;

    // Spawn the connection handler
    tokio::spawn(async move {
        if let Err(e) = connection.await {
            eprintln!("  ⚠️ PostgreSQL connection error: {}", e);
        }
    });

    // Create table
    pg_client
        .execute(
            "CREATE TABLE texts (id SERIAL PRIMARY KEY, text TEXT NOT NULL)",
            &[],
        )
        .await
        .context("Failed to create table")?;
    println!("  ✓ Created table 'texts'");

    // Bulk insert via COPY protocol
    println!("  ℹ️ Bulk inserting rows via COPY...");
    let insert_start = Instant::now();
    let mut rows_inserted = 0usize;

    let reader = ShardedParquetReader::new(shard_paths.clone(), cfg.column.to_string(), num_rows);

    for batch_result in reader {
        let batch = batch_result?;
        let col = batch.column(0);
        let texts = extract_text_values(col.as_ref());

        // Build COPY data buffer
        let mut copy_data = Vec::new();
        for text_opt in &texts {
            match text_opt {
                Some(text) => {
                    copy_data.extend_from_slice(escape_copy_text(text).as_bytes());
                    copy_data.push(b'\n');
                }
                None => {
                    // Skip nulls — our table is NOT NULL
                    continue;
                }
            }
        }

        let sink = pg_client
            .copy_in("COPY texts(text) FROM STDIN")
            .await
            .context("Failed to start COPY")?;
        futures::pin_mut!(sink);
        sink.send(bytes::Bytes::from(copy_data)).await?;
        sink.close().await?;

        rows_inserted += texts.iter().filter(|t| t.is_some()).count();

        if rows_inserted % 500_000 < 10_000 {
            println!(
                "    [{:.1}s] Inserted {} rows...",
                insert_start.elapsed().as_secs_f64(),
                rows_inserted,
            );
        }
    }

    println!(
        "  ✓ Inserted {} rows in {:.1}s",
        rows_inserted,
        insert_start.elapsed().as_secs_f64(),
    );

    // Build GIN FTS index with polling monitor
    println!("  ℹ️ Creating GIN FTS index...");

    let disk_paths = vec![pg_data_dir.clone(), PathBuf::from("/tmp")];
    let monitor = PollingMonitor::new();
    let mut rss_monitor = PeakRssMonitor::new();
    rss_monitor.start();

    let index_start = Instant::now();
    let poll_handle = monitor.start(disk_paths, index_start);

    pg_client
        .execute(
            "CREATE INDEX idx_texts_fts ON texts USING GIN(to_tsvector('english', text))",
            &[],
        )
        .await
        .context("Failed to create GIN FTS index")?;

    let duration_ns = index_start.elapsed().as_nanos() as u64;
    let samples = monitor.stop_and_take();
    poll_handle.await?;
    let (peak_rss, delta_rss) = rss_monitor.stop();

    let duration_s = duration_ns as f64 / 1_000_000_000.0;
    println!("  ✓ GIN index created in {:.2}s", duration_s);
    println!(
        "  ✓ Peak RSS: {:.0} MB, delta RSS: {:.0} MB",
        peak_rss as f64 / 1_000_000.0,
        delta_rss as f64 / 1_000_000.0,
    );

    // Stop PostgreSQL
    println!("  ℹ️ Stopping PostgreSQL...");
    postgresql
        .stop()
        .await
        .context("Failed to stop PostgreSQL")?;
    println!("  ✓ PostgreSQL stopped");

    Ok((
        BenchmarkResult {
            benchmark_name: bench_name,
            dataset_name,
            dataset_description: cfg.description.to_string(),
            num_rows: rows_inserted,
            total_text_bytes,
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
    env_logger::init();

    let args = Args::parse();

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

    // Ensure cache directory exists
    std::fs::create_dir_all(&args.cache_dir)?;

    println!("ℹ️ FTS Index Training Benchmark — PostgreSQL (GIN)");
    println!("  Datasets: {:?}", dataset_keys);
    println!("  Cache dir: {}", args.cache_dir.display());

    let client = reqwest::Client::builder()
        .user_agent("lance-bench/0.1.0")
        .build()?;

    let mut results = Vec::new();
    let mut all_samples: Vec<(String, ProgressSample)> = Vec::new();

    for &key in &dataset_keys {
        let cfg = DATASETS.iter().find(|d| d.key == key).unwrap();
        let num_rows = args.num_rows.unwrap_or(cfg.default_rows);

        let (result, samples) = run_benchmark(
            cfg,
            num_rows,
            &args.cache_dir,
            args.force_recreate,
            &client,
        )
        .await?;
        let bench_name = result.benchmark_name.clone();
        results.push(result);
        all_samples.extend(samples.into_iter().map(|s| (bench_name.clone(), s)));
    }

    // Write JSON output
    let output = BenchmarkOutput {
        benchmark_type: "train_fts_index_postgres".to_string(),
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
            "benchmark_name,elapsed_s,rss_bytes,rss_mb,cpu_active,disk_delta_bytes,disk_delta_mb"
        )?;
        for (name, s) in &all_samples {
            writeln!(
                file,
                "{},{:.3},{},{:.1},{:.2},{},{:.1}",
                name,
                s.elapsed_s,
                s.rss_bytes,
                s.rss_bytes as f64 / 1_000_000.0,
                s.cpu_active,
                s.disk_delta_bytes,
                s.disk_delta_bytes as f64 / 1_000_000.0,
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
