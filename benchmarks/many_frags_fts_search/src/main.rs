//! Many-Fragments FTS Search Benchmark
//!
//! Measures FTS search latency when a Lance dataset has many small unindexed
//! fragments. This simulates real-world scenarios where data has been appended
//! since the last index build.
//!
//! The benchmark creates a base dataset, indexes it, appends many small
//! unindexed fragments, then times FTS queries.

use anyhow::{Context, Result};
use arrow_array::{FixedSizeBinaryArray, RecordBatch, RecordBatchIterator, StringArray};
use arrow_schema::{DataType, Field, Schema};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use lance::Dataset;
use lance_index::scalar::{FullTextSearchQuery, InvertedIndexParams};
use lance_index::{DatasetIndexExt, IndexType};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

extern crate jemallocator;

#[global_allocator]
static GLOBAL: jemallocator::Jemalloc = jemallocator::Jemalloc;

// ---------------------------------------------------------------------------
// Word pool for text generation
// ---------------------------------------------------------------------------

const WORD_POOL: &[&str] = &[
    "the",
    "be",
    "to",
    "of",
    "and",
    "a",
    "in",
    "that",
    "have",
    "I",
    "it",
    "for",
    "not",
    "on",
    "with",
    "he",
    "as",
    "you",
    "do",
    "at",
    "this",
    "but",
    "his",
    "by",
    "from",
    "they",
    "we",
    "say",
    "her",
    "she",
    "or",
    "an",
    "will",
    "my",
    "one",
    "all",
    "would",
    "there",
    "their",
    "what",
    "so",
    "up",
    "out",
    "if",
    "about",
    "who",
    "get",
    "which",
    "go",
    "me",
    "when",
    "make",
    "can",
    "like",
    "time",
    "no",
    "just",
    "him",
    "know",
    "take",
    "people",
    "into",
    "year",
    "your",
    "good",
    "some",
    "could",
    "them",
    "see",
    "other",
    "than",
    "then",
    "now",
    "look",
    "only",
    "come",
    "its",
    "over",
    "think",
    "also",
    "back",
    "after",
    "use",
    "two",
    "how",
    "our",
    "work",
    "first",
    "well",
    "way",
    "even",
    "new",
    "want",
    "because",
    "any",
    "these",
    "give",
    "day",
    "most",
    "us",
    "great",
    "between",
    "need",
    "large",
    "often",
    "important",
    "long",
    "thing",
    "own",
    "point",
    "provide",
    "different",
    "place",
    "while",
    "high",
    "right",
    "might",
    "still",
    "begin",
    "life",
    "country",
    "help",
    "world",
    "school",
    "every",
    "never",
    "next",
    "below",
    "last",
    "ask",
    "found",
    "home",
    "state",
    "move",
    "where",
    "show",
    "always",
    "student",
    "again",
    "change",
    "each",
    "around",
    "follow",
    "under",
    "keep",
    "last",
    "read",
    "hand",
    "few",
    "small",
    "number",
    "part",
    "turn",
    "real",
    "leave",
    "history",
    "city",
    "much",
    "early",
    "open",
    "seem",
    "together",
    "group",
    "run",
    "start",
    "develop",
    "system",
    "process",
    "set",
    "story",
    "fact",
    "industry",
    "data",
    "search",
    "research",
    "information",
    "power",
    "learn",
    "question",
    "area",
    "problem",
    "result",
    "report",
    "level",
    "order",
    "program",
    "action",
    "company",
    "market",
    "service",
    "human",
    "local",
    "social",
    "general",
    "public",
    "include",
    "quite",
    "example",
    "create",
    "study",
];

/// Generate a random text string of approximately `target_bytes` length.
fn generate_text(rng: &mut StdRng, target_bytes: usize) -> String {
    let mut text = String::with_capacity(target_bytes + 32);
    while text.len() < target_bytes {
        if !text.is_empty() {
            text.push(' ');
        }
        let word = WORD_POOL[rng.gen_range(0..WORD_POOL.len())];
        text.push_str(word);
    }
    text.truncate(target_bytes);
    text
}

// ---------------------------------------------------------------------------
// Batch generation
// ---------------------------------------------------------------------------

const BATCH_SIZE: usize = 10_000;
const TEXT_BYTES_PER_ROW: usize = 2048;

fn dataset_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("text", DataType::Utf8, false),
        Field::new("uuid", DataType::FixedSizeBinary(16), false),
    ]))
}

/// Iterator that yields RecordBatches of generated text + uuid data.
struct TextBatchGenerator {
    rng: StdRng,
    remaining: usize,
    schema: Arc<Schema>,
}

impl TextBatchGenerator {
    fn new(total_rows: usize, seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
            remaining: total_rows,
            schema: dataset_schema(),
        }
    }

    fn generate_batch(&mut self, num_rows: usize) -> RecordBatch {
        let mut texts = Vec::with_capacity(num_rows);
        let mut uuids: Vec<[u8; 16]> = Vec::with_capacity(num_rows);

        for _ in 0..num_rows {
            texts.push(generate_text(&mut self.rng, TEXT_BYTES_PER_ROW));
            let mut uuid = [0u8; 16];
            self.rng.fill(&mut uuid);
            uuids.push(uuid);
        }

        let text_array = StringArray::from(texts);
        let uuid_refs: Vec<&[u8]> = uuids.iter().map(|u| u.as_slice()).collect();
        let uuid_array = FixedSizeBinaryArray::try_from_iter(uuid_refs.into_iter()).unwrap();

        RecordBatch::try_new(
            self.schema.clone(),
            vec![Arc::new(text_array), Arc::new(uuid_array)],
        )
        .unwrap()
    }
}

impl Iterator for TextBatchGenerator {
    type Item = std::result::Result<RecordBatch, arrow_schema::ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let batch_rows = self.remaining.min(BATCH_SIZE);
        self.remaining -= batch_rows;
        Some(Ok(self.generate_batch(batch_rows)))
    }
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser, Debug)]
#[command(name = "many-frags-fts-search-benchmark")]
#[command(about = "Benchmark FTS search latency with many small unindexed fragments")]
struct Args {
    /// Number of rows in the base dataset
    #[arg(long, default_value_t = 1_000_000)]
    num_rows: usize,

    /// Number of extra unindexed fragments to append
    #[arg(long, default_value_t = 500)]
    num_extra_fragments: usize,

    /// Rows per extra fragment
    #[arg(long, default_value_t = 10)]
    rows_per_fragment: usize,

    /// Number of search iterations
    #[arg(long, default_value_t = 10)]
    iterations: usize,

    /// FTS search query term
    #[arg(long, default_value = "history")]
    query: String,

    /// JSON output path
    #[arg(long, default_value = "many-frags-fts-search-results.json")]
    output: PathBuf,

    /// Cache directory for generated datasets
    #[arg(long, default_value_os_t = default_cache_dir())]
    cache_dir: PathBuf,

    /// Force re-creation of cached dataset
    #[arg(long)]
    force_recreate: bool,

    /// Run a single iteration and print the analyze plan instead of benchmarking
    #[arg(long)]
    analyze: bool,

    /// Enable Chrome trace event output (viewable in chrome://tracing or Perfetto)
    #[arg(long)]
    chrome_trace: Option<PathBuf>,
}

fn default_cache_dir() -> PathBuf {
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    home.join(".cache/lance-bench/many-frags-fts-search")
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
    num_rows: usize,
    num_extra_fragments: usize,
    rows_per_fragment: usize,
    query: String,
    iterations: usize,
    duration_ns: u64,
    min_ns: u64,
    max_ns: u64,
    values_ns: Vec<u64>,
}

// ---------------------------------------------------------------------------
// Dataset metadata (for caching)
// ---------------------------------------------------------------------------

#[derive(Serialize, serde::Deserialize)]
struct DatasetMeta {
    num_rows: usize,
    num_extra_fragments: usize,
    rows_per_fragment: usize,
}

// ---------------------------------------------------------------------------
// Dataset preparation
// ---------------------------------------------------------------------------

async fn ensure_dataset(
    num_rows: usize,
    num_extra_fragments: usize,
    rows_per_fragment: usize,
    cache_dir: &Path,
    force_recreate: bool,
) -> Result<PathBuf> {
    let dataset_name = format!("{num_rows}_{num_extra_fragments}_{rows_per_fragment}");
    let lance_path = cache_dir.join(format!("{dataset_name}.lance"));
    let meta_path = cache_dir.join(format!("{dataset_name}.meta.json"));

    // Check cache
    if !force_recreate && lance_path.exists() && meta_path.exists() {
        let meta: DatasetMeta = serde_json::from_str(&std::fs::read_to_string(&meta_path)?)?;
        if meta.num_rows == num_rows
            && meta.num_extra_fragments == num_extra_fragments
            && meta.rows_per_fragment == rows_per_fragment
        {
            println!(
                "  ✓ Reusing cached dataset ({} rows + {} fragments x {} rows)",
                meta.num_rows, meta.num_extra_fragments, meta.rows_per_fragment,
            );
            return Ok(lance_path);
        }
    }

    // Clean up existing dataset if present
    if lance_path.exists() {
        std::fs::remove_dir_all(&lance_path)?;
    }

    // Step 1: Generate base dataset
    println!("  ℹ️ Generating base dataset ({num_rows} rows)...");
    let generator = TextBatchGenerator::new(num_rows, 42);
    let schema = dataset_schema();
    let reader = RecordBatchIterator::new(generator, schema);

    Dataset::write(reader, lance_path.to_str().unwrap(), None).await?;
    println!("  ✓ Base dataset written");

    // Step 2: Create FTS inverted index
    println!("  ℹ️ Creating FTS inverted index...");
    let mut dataset = Dataset::open(lance_path.to_str().unwrap()).await?;
    dataset
        .create_index_builder(
            &["text"],
            IndexType::Inverted,
            &InvertedIndexParams::default(),
        )
        .replace(true)
        .await
        .context("Failed to create FTS index")?;
    println!("  ✓ FTS index created");

    // Step 3: Append many small fragments
    println!("  ℹ️ Appending {num_extra_fragments} fragments ({rows_per_fragment} rows each)...");
    let pb = ProgressBar::new(num_extra_fragments as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("    [{bar:40}] {pos}/{len} fragments ({eta} remaining)")
            .unwrap(),
    );

    // Use a different seed range for extra fragments so they don't duplicate base data
    let mut frag_rng = StdRng::seed_from_u64(12345);
    for _ in 0..num_extra_fragments {
        let frag_seed = frag_rng.gen::<u64>();
        let mut gen = TextBatchGenerator::new(rows_per_fragment, frag_seed);
        let batch = gen.generate_batch(rows_per_fragment);
        let schema = dataset_schema();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
        dataset.append(reader, None).await?;
        pb.inc(1);
    }
    pb.finish_and_clear();
    println!(
        "  ✓ Appended {num_extra_fragments} fragments ({} extra rows total)",
        num_extra_fragments * rows_per_fragment,
    );

    // Write cache metadata
    let meta = DatasetMeta {
        num_rows,
        num_extra_fragments,
        rows_per_fragment,
    };
    std::fs::write(&meta_path, serde_json::to_string(&meta)?)?;

    Ok(lance_path)
}

// ---------------------------------------------------------------------------
// Benchmark execution
// ---------------------------------------------------------------------------

async fn run_analyze(args: &Args) -> Result<()> {
    println!("\n{}", "=".repeat(60));
    println!("  Analyze Plan");
    println!("{}", "=".repeat(60));

    // Ensure dataset exists
    let lance_path = ensure_dataset(
        args.num_rows,
        args.num_extra_fragments,
        args.rows_per_fragment,
        &args.cache_dir,
        args.force_recreate,
    )
    .await?;

    // Open dataset
    let dataset = Dataset::open(lance_path.to_str().unwrap()).await?;
    println!(
        "  ℹ️ Dataset opened: {} fragments",
        dataset.count_fragments(),
    );

    // Build scanner with FTS query and get analyze plan
    let query = FullTextSearchQuery::new(args.query.clone()).with_column("text".to_string())?;
    let mut scanner = dataset.scan();
    scanner.full_text_search(query)?.limit(Some(50), None)?;

    println!(
        "  ℹ️ Running analyze plan for query \"{}\"...\n",
        args.query
    );
    let plan = scanner.analyze_plan().await?;
    println!("{}", plan);

    Ok(())
}

async fn run_benchmark(args: &Args) -> Result<BenchmarkResult> {
    let bench_name = format!(
        "many_frags_fts_search/rows={}/fragments={}/rows_per_frag={}/query={}",
        args.num_rows, args.num_extra_fragments, args.rows_per_fragment, args.query,
    );

    println!("\n{}", "=".repeat(60));
    println!("  Benchmark: {}", bench_name);
    println!("{}", "=".repeat(60));

    // Ensure dataset exists
    let lance_path = ensure_dataset(
        args.num_rows,
        args.num_extra_fragments,
        args.rows_per_fragment,
        &args.cache_dir,
        args.force_recreate,
    )
    .await?;

    // Open dataset
    let dataset = Dataset::open(lance_path.to_str().unwrap()).await?;
    println!(
        "  ℹ️ Dataset opened: {} fragments",
        dataset.count_fragments(),
    );

    // Warmup: run one search and discard
    println!("  ℹ️ Warming up...");
    {
        let query = FullTextSearchQuery::new(args.query.clone()).with_column("text".to_string())?;
        let mut scanner = dataset.scan();
        scanner.full_text_search(query)?;
        let batch = scanner.try_into_batch().await?;
        println!("  ✓ Warmup complete ({} rows returned)", batch.num_rows());
    }

    tracing::info!("Running benchmark iterations");

    // Run benchmark iterations
    println!("  ℹ️ Running {} iterations...", args.iterations);
    let mut timings_ns = Vec::with_capacity(args.iterations);
    let mut last_row_count = 0;

    for i in 0..args.iterations {
        let query = FullTextSearchQuery::new(args.query.clone()).with_column("text".to_string())?;
        let mut scanner = dataset.scan();
        scanner.full_text_search(query)?.limit(Some(50), None)?;

        let start = Instant::now();
        let batch = scanner.try_into_batch().await?;
        let elapsed_ns = start.elapsed().as_nanos() as u64;

        last_row_count = batch.num_rows();
        timings_ns.push(elapsed_ns);
        println!(
            "    Iteration {}/{}: {:.2}ms ({} rows)",
            i + 1,
            args.iterations,
            elapsed_ns as f64 / 1_000_000.0,
            last_row_count,
        );
    }

    let min_ns = *timings_ns.iter().min().unwrap();
    let max_ns = *timings_ns.iter().max().unwrap();
    let mean_ns = timings_ns.iter().sum::<u64>() / timings_ns.len() as u64;

    println!("\n  Results:");
    println!("    Min:  {:.2}ms", min_ns as f64 / 1_000_000.0);
    println!("    Max:  {:.2}ms", max_ns as f64 / 1_000_000.0);
    println!("    Mean: {:.2}ms", mean_ns as f64 / 1_000_000.0);
    println!("    Rows returned: {}", last_row_count);

    Ok(BenchmarkResult {
        benchmark_name: bench_name,
        num_rows: args.num_rows,
        num_extra_fragments: args.num_extra_fragments,
        rows_per_fragment: args.rows_per_fragment,
        query: args.query.clone(),
        iterations: args.iterations,
        duration_ns: mean_ns,
        min_ns,
        max_ns,
        values_ns: timings_ns,
    })
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
        println!("ℹ️ Chrome trace output enabled: {}", trace_path.display());
        Some(guard)
    } else {
        env_logger::init();
        None
    };

    println!("ℹ️ Many-Fragments FTS Search Benchmark");
    println!("  Base rows:        {}", args.num_rows);
    println!("  Extra fragments:  {}", args.num_extra_fragments);
    println!("  Rows per fragment: {}", args.rows_per_fragment);
    println!("  Query:            \"{}\"", args.query);
    println!("  Iterations:       {}", args.iterations);
    println!("  Cache dir:        {}", args.cache_dir.display());

    // Ensure cache directory exists
    std::fs::create_dir_all(&args.cache_dir)?;

    if args.analyze {
        run_analyze(&args).await?;
        return Ok(());
    }

    let result = run_benchmark(&args).await?;

    // Write JSON output
    let output = BenchmarkOutput {
        benchmark_type: "many_frags_fts_search".to_string(),
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs(),
        results: vec![result],
    };

    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&args.output, serde_json::to_string_pretty(&output)?)?;

    println!("\n✓ Results written to {}", args.output.display());

    Ok(())
}
