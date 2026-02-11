//! Scan Benchmark
//!
//! Benchmarks full table scan performance across different storage engines.
//!
//! Supports:
//! - Lance
//! - Parquet (sync)
//! - Parquet (async)
//! - Vortex

use anyhow::Result;
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

mod cache;
mod engines;
mod input;
mod stats;

use engines::{create_registry, ScanEngine, ScanHandle};
use stats::compute_statistics;

extern crate jemallocator;

#[global_allocator]
static GLOBAL: jemallocator::Jemalloc = jemallocator::Jemalloc;

/// Scan benchmark configuration.
#[derive(Parser, Debug, Clone)]
#[command(name = "scan-benchmark")]
#[command(about = "Benchmark full table scan performance across storage engines")]
pub struct Config {
    /// Input file path (format detected from extension: .csv, .parquet, .json, .lance)
    #[arg(short, long)]
    pub input: String,

    /// Engines to benchmark (comma-separated, or "all")
    #[arg(short, long, default_value = "all")]
    pub engines: String,

    /// Output directory for converted files
    #[arg(short, long, default_value = "/tmp/scan-benchmark")]
    pub output_dir: String,

    /// Number of timed iterations
    #[arg(long, default_value_t = 10)]
    pub iterations: usize,

    /// Number of warmup iterations
    #[arg(long, default_value_t = 2)]
    pub warmup_iterations: usize,

    /// Skip warmup phase
    #[arg(long, default_value_t = false)]
    pub skip_warmup: bool,

    /// Skip cache drop between warmup and timed phase
    #[arg(long, default_value_t = false)]
    pub skip_cache_drop: bool,
}

/// Results for a single engine benchmark.
struct EngineResult {
    name: String,
    file_size: u64,
    row_count: usize,
    latencies: Vec<f64>,
}

fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

fn format_throughput(bytes: u64, seconds: f64) -> String {
    let bytes_per_sec = bytes as f64 / seconds;
    format_bytes(bytes_per_sec as u64) + "/s"
}

fn format_rows_per_sec(rows: usize, seconds: f64) -> String {
    let rps = rows as f64 / seconds;
    if rps >= 1_000_000.0 {
        format!("{:.2}M rows/s", rps / 1_000_000.0)
    } else if rps >= 1_000.0 {
        format!("{:.2}K rows/s", rps / 1_000.0)
    } else {
        format!("{:.0} rows/s", rps)
    }
}

/// Result of a single scan operation.
struct ScanResult {
    latency: f64,
    rows_scanned: usize,
}

async fn run_scan(handle: Arc<dyn ScanHandle>) -> Result<ScanResult> {
    let start = Instant::now();
    let batches = handle.scan().await?;
    let latency = start.elapsed().as_secs_f64();

    // Count actual rows scanned
    let rows_scanned: usize = batches.iter().map(|b| b.num_rows()).sum();

    Ok(ScanResult { latency, rows_scanned })
}

fn benchmark_engine(
    engine: Arc<dyn ScanEngine>,
    handle: Arc<dyn ScanHandle>,
    uri: &str,
    config: &Config,
    expected_rows: usize,
) -> Result<EngineResult> {
    let runtime = engine.runtime();

    // Verify row count from handle metadata
    let handle_row_count = handle.row_count();
    if handle_row_count != expected_rows {
        anyhow::bail!(
            "Row count mismatch for {}: handle reports {} rows, expected {}",
            engine.name(),
            handle_row_count,
            expected_rows
        );
    }

    // Warmup phase
    if !config.skip_warmup && config.warmup_iterations > 0 {
        let pb = ProgressBar::new(config.warmup_iterations as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("    Warmup [{bar:30}] {pos}/{len}")
                .unwrap(),
        );

        for _ in 0..config.warmup_iterations {
            let result = runtime.block_on(run_scan(handle.clone()))?;
            if result.rows_scanned != expected_rows {
                anyhow::bail!(
                    "Warmup scan row count mismatch for {}: scanned {} rows, expected {}",
                    engine.name(),
                    result.rows_scanned,
                    expected_rows
                );
            }
            pb.inc(1);
        }
        pb.finish();
    }

    // Drop cache
    if !config.skip_cache_drop {
        print!("    Dropping cache... ");
        engine.drop_cache(uri)?;
        println!("done");
    }

    // Timed phase
    let pb = ProgressBar::new(config.iterations as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("    Timed  [{bar:30}] {pos}/{len}")
            .unwrap(),
    );

    let mut latencies = Vec::with_capacity(config.iterations);
    let mut total_rows_scanned = 0usize;
    for _ in 0..config.iterations {
        let result = runtime.block_on(run_scan(handle.clone()))?;
        if result.rows_scanned != expected_rows {
            anyhow::bail!(
                "Timed scan row count mismatch for {}: scanned {} rows, expected {}",
                engine.name(),
                result.rows_scanned,
                expected_rows
            );
        }
        total_rows_scanned += result.rows_scanned;
        latencies.push(result.latency);
        pb.inc(1);
    }
    pb.finish();

    println!("    Verified: {} rows per scan ({} total)", expected_rows, total_rows_scanned);

    Ok(EngineResult {
        name: engine.name().to_string(),
        file_size: handle.byte_size(),
        row_count: handle.row_count(),
        latencies,
    })
}

fn print_result(result: &EngineResult, input_size: u64) {
    let stats = compute_statistics(&result.latencies);

    let compression_ratio = result.file_size as f64 / input_size as f64;

    println!("\n  File size: {} ({:.2}x input)", format_bytes(result.file_size), compression_ratio);
    println!("  Row count: {}", result.row_count);
    println!();
    println!("  Latency (seconds):");
    println!("    mean:   {:.4}", stats.mean);
    println!("    std:    {:.4}", stats.std);
    println!("    min:    {:.4}", stats.min);
    println!("    max:    {:.4}", stats.max);
    println!("    p50:    {:.4}", stats.p50);
    println!("    p95:    {:.4}", stats.p95);
    println!("    p99:    {:.4}", stats.p99);
    println!();
    println!(
        "  Throughput: {}, {}",
        format_throughput(result.file_size, stats.mean),
        format_rows_per_sec(result.row_count, stats.mean)
    );
}

fn print_comparison(results: &[EngineResult]) {
    if results.len() < 2 {
        return;
    }

    println!("\n{}", "=".repeat(70));
    println!("COMPARISON SUMMARY");
    println!("{}", "=".repeat(70));

    // Find fastest (lowest mean latency)
    let fastest = results
        .iter()
        .min_by(|a, b| {
            let mean_a: f64 = a.latencies.iter().sum::<f64>() / a.latencies.len() as f64;
            let mean_b: f64 = b.latencies.iter().sum::<f64>() / b.latencies.len() as f64;
            mean_a.partial_cmp(&mean_b).unwrap()
        })
        .unwrap();
    let fastest_mean: f64 = fastest.latencies.iter().sum::<f64>() / fastest.latencies.len() as f64;

    // Find smallest file
    let smallest = results.iter().min_by_key(|r| r.file_size).unwrap();

    println!("\n  {:20} {:>12} {:>12} {:>12}", "Engine", "Mean (s)", "vs Fastest", "File Size");
    println!("  {}", "-".repeat(60));

    for result in results {
        let mean = result.latencies.iter().sum::<f64>() / result.latencies.len() as f64;
        let vs_fastest = mean / fastest_mean;
        println!(
            "  {:20} {:>12.4} {:>11.2}x {:>12}",
            result.name,
            mean,
            vs_fastest,
            format_bytes(result.file_size)
        );
    }

    println!();
    println!("  Fastest: {} ({:.4}s mean)", fastest.name, fastest_mean);
    println!("  Smallest: {} ({})", smallest.name, format_bytes(smallest.file_size));
}

fn main() -> Result<()> {
    env_logger::init();

    let config = Config::parse();

    // Validate input file exists
    let input_path = Path::new(&config.input);
    if !input_path.exists() {
        anyhow::bail!("Input file does not exist: {}", config.input);
    }

    let input_size = input_path.metadata()?.len();

    println!("{}", "=".repeat(70));
    println!("Scan Benchmark");
    println!("{}", "=".repeat(70));

    println!("\nConfiguration:");
    println!("  Input file: {}", config.input);
    println!("  Input size: {}", format_bytes(input_size));
    println!("  Output directory: {}", config.output_dir);
    println!("  Iterations: {} (+ {} warmup)", config.iterations, config.warmup_iterations);

    // Step 1: Load input file
    println!("\n{}", "=".repeat(70));
    println!("Step 1: Loading Input File");
    println!("{}", "=".repeat(70));

    let start = Instant::now();
    let batches = input::load_input(input_path)?;
    let load_time = start.elapsed();

    let total_rows = input::total_rows(&batches);
    let total_bytes = input::total_bytes(&batches);
    println!("  Loaded {} batches, {} rows, {} in memory", batches.len(), total_rows, format_bytes(total_bytes as u64));
    println!("  Load time: {:.2}s", load_time.as_secs_f64());

    // Step 2: Determine engines to benchmark
    let registry = create_registry();
    let engine_names: Vec<String> = if config.engines.to_lowercase() == "all" {
        registry.available().iter().map(|s| s.to_string()).collect()
    } else {
        config.engines.split(',').map(|s| s.trim().to_string()).collect()
    };

    println!("\n  Engines to benchmark: {:?}", engine_names);

    // Step 3: Write data to each engine format and benchmark
    println!("\n{}", "=".repeat(70));
    println!("Step 2: Benchmarking Engines");
    println!("{}", "=".repeat(70));

    let mut results = Vec::new();

    for engine_name in &engine_names {
        let engine = match registry.get(engine_name) {
            Some(e) => e,
            None => {
                eprintln!("  Warning: Unknown engine '{}', skipping", engine_name);
                continue;
            }
        };

        println!("\n--- {} ---", engine.name().to_uppercase());

        let uri = format!("{}/{}", config.output_dir, engine.name());

        // Write data if not exists
        let handle = if engine.exists(&uri) {
            println!("  Dataset exists, opening...");
            engine.open(&uri)?
        } else {
            print!("  Writing dataset... ");
            let start = Instant::now();
            let handle = engine.write(&uri, &batches)?;
            let write_time = start.elapsed().as_secs_f64();
            let written_rows = handle.row_count();
            println!("done ({:.2}s, {} rows written)", write_time, written_rows);

            // Verify row count matches input
            if written_rows != total_rows {
                anyhow::bail!(
                    "Write verification failed for {}: wrote {} rows, expected {}",
                    engine.name(),
                    written_rows,
                    total_rows
                );
            }
            handle
        };

        println!("  Running benchmark...");
        let result = benchmark_engine(engine, handle, &uri, &config, total_rows)?;
        print_result(&result, input_size);
        results.push(result);
    }

    // Step 4: Print comparison
    print_comparison(&results);

    println!("\n{}", "=".repeat(70));
    println!("Benchmark Complete!");
    println!("{}", "=".repeat(70));

    Ok(())
}
