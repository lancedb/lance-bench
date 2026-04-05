//! Plan Executor Throughput Benchmark
//!
//! Measures take throughput (takes/second) by running a real plan executor server
//! and issuing RandomTake gRPC requests against it.
//!
//! Dataset schema:
//!   - id: UInt64
//!   - vector: FixedSizeList[Float32; 1024]  (4 KiB per row)
//!   - str_payload_0..3: Utf8  (12-16 random characters)
//!   - float_payload_0..3: Float64
//!
//! Dataset layout: 1024 fragments, 16384 rows per fragment (16,777,216 total rows).
//!
//! Cache modes:
//!   - none:   No page cache — every read hits object store.
//!   - disk:   Read-through disk page cache in front of object store.
//!   - memory: In-memory page cache in front of object store.

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use arrow::array::{Array, Float64Array, RecordBatch, StringBuilder, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use clap::Parser;
use hyper_util::rt::TokioIo;
use indicatif::{ProgressBar, ProgressStyle};
use lance::dataset::{Dataset, WriteMode, WriteParams};
use lance_encoding::version::LanceFileVersion;
use plan_executor::bootstrap::ServerParams;
use plan_executor::executor::DefaultPlanExecutor;
use prometheus::Registry;
use rand::distributions::Alphanumeric;
use rand::Rng;
use serde::Serialize;
use sophon_caching::object_store::config::{
    ChainedCachesConfig, InMemoryCacheEngine, ObjectStoreCacheConfig,
};
use sophon_protos::remote_plan_worker::execute_plan_request::Plan;
use sophon_protos::remote_plan_worker::remote_plan_worker_client::RemotePlanWorkerClient;
use sophon_protos::remote_plan_worker::{
    ExecutePlanRequest, PageCacheExecutorStatusRequest, PreWarmCache, RandomTake,
};
use tempfile::TempDir;
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::Semaphore;
use tokio_stream::wrappers::UnixListenerStream;
use tokio_util::sync::CancellationToken;
use tonic::transport::{Channel, Endpoint, Server, Uri};
use tower::service_fn;

#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

// ---------------------------------------------------------------------------
// Prometheus metrics helpers
// ---------------------------------------------------------------------------

fn get_counter_value(registry: &Registry, name: &str) -> u64 {
    for mf in registry.gather() {
        if mf.get_name() == name {
            let metrics = mf.get_metric();
            if let Some(m) = metrics.first() {
                return m.get_counter().get_value() as u64;
            }
        }
    }
    0
}

/// Sum all read-related object store call counters.
fn get_total_read_iops(registry: &Registry) -> u64 {
    get_counter_value(registry, "object_store_get_calls_total")
        + get_counter_value(registry, "object_store_get_opts_calls_total")
        + get_counter_value(registry, "object_store_get_range_calls_total")
        + get_counter_value(registry, "object_store_get_ranges_calls_total")
}

/// Parse a counter value from the Prometheus exposition format rendered
/// by `metrics_exporter_prometheus::PrometheusHandle::render()`.
fn get_metrics_counter(rendered: &str, name: &str) -> u64 {
    for line in rendered.lines() {
        if line.starts_with(name) && !line.starts_with('#') {
            // Line format: "metric_name value" or "metric_name{labels} value"
            if let Some(val_str) = line.rsplit(' ').next() {
                if let Ok(val) = val_str.parse::<f64>() {
                    return val as u64;
                }
            }
        }
    }
    0
}

fn get_total_read_bytes(registry: &Registry) -> u64 {
    get_counter_value(registry, "object_store_get_bytes_total")
        + get_counter_value(registry, "object_store_get_opts_bytes_total")
        + get_counter_value(registry, "object_store_get_range_bytes_total")
        + get_counter_value(registry, "object_store_get_ranges_bytes_total")
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, clap::ValueEnum)]
enum CacheMode {
    /// No page cache — all reads hit object store directly.
    None,
    /// Read-through disk page cache in front of object store.
    Disk,
    /// In-memory page cache in front of object store.
    Memory,
}

#[derive(Debug, Clone, clap::ValueEnum)]
enum ColumnSet {
    /// Take only the id column.
    IdOnly,
    /// Take id + string payload columns.
    IdAndStrings,
    /// Take id + float payload columns.
    IdAndFloats,
    /// Take id + all payload columns (no vector).
    IdAndPayloads,
    /// Take all columns including vector.
    All,
}

impl ColumnSet {
    fn column_names(&self) -> Vec<&'static str> {
        let mut cols = vec!["id"];
        match self {
            ColumnSet::IdOnly => {}
            ColumnSet::IdAndStrings => {
                cols.extend(["str_payload_0", "str_payload_1", "str_payload_2", "str_payload_3"]);
            }
            ColumnSet::IdAndFloats => {
                cols.extend([
                    "float_payload_0",
                    "float_payload_1",
                    "float_payload_2",
                    "float_payload_3",
                ]);
            }
            ColumnSet::IdAndPayloads => {
                cols.extend([
                    "str_payload_0",
                    "str_payload_1",
                    "str_payload_2",
                    "str_payload_3",
                    "float_payload_0",
                    "float_payload_1",
                    "float_payload_2",
                    "float_payload_3",
                ]);
            }
            ColumnSet::All => {
                cols.extend([
                    "vector",
                    "str_payload_0",
                    "str_payload_1",
                    "str_payload_2",
                    "str_payload_3",
                    "float_payload_0",
                    "float_payload_1",
                    "float_payload_2",
                    "float_payload_3",
                ]);
            }
        }
        cols
    }
}

#[derive(Parser, Debug)]
#[command(name = "pe-throughput-benchmark")]
#[command(about = "Benchmark plan executor take throughput")]
struct Args {
    /// Dataset URI.
    #[arg(long, default_value = "s3://weston-s3-lance-test/pe-throughput")]
    dataset_uri: String,

    /// Number of fragments in the dataset.
    #[arg(long, default_value_t = 1024)]
    num_fragments: u32,

    /// Rows per fragment.
    #[arg(long, default_value_t = 16384)]
    rows_per_fragment: u32,

    /// Vector dimension (number of f32 elements). 1024 = 4 KiB per vector.
    #[arg(long, default_value_t = 1024)]
    vector_dim: usize,

    /// Number of row ids per take request.
    #[arg(long, default_value_t = 100)]
    take_size: usize,

    /// Total number of take requests to issue.
    #[arg(long, default_value_t = 1000)]
    num_takes: usize,

    /// Maximum number of concurrent take requests in flight.
    #[arg(long, default_value_t = 16)]
    concurrency: usize,

    /// Which columns to include in the take projection.
    #[arg(long, value_enum, default_value_t = ColumnSet::IdOnly)]
    columns: ColumnSet,

    /// Cache mode.
    #[arg(long, value_enum, default_value_t = CacheMode::None)]
    cache_mode: CacheMode,

    /// Disk cache directory (used when cache_mode is disk).
    #[arg(long)]
    disk_cache_path: Option<String>,

    /// Cache capacity in MB.
    #[arg(long, default_value_t = 512)]
    cache_size_mb: u64,

    /// Skip dataset creation (assume it already exists).
    #[arg(long, default_value_t = false)]
    skip_create: bool,

    /// Number of warmup takes before the timed run (skipped if --prewarm is set).
    #[arg(long, default_value_t = 50)]
    warmup_takes: usize,

    /// Use the plan executor's PreWarmCache to populate the cache before the
    /// timed run, instead of warmup takes.
    #[arg(long, default_value_t = false)]
    prewarm: bool,

    /// Limit takes and prewarm to the first N fragments. If not set, all
    /// fragments are used. Useful for fitting the working set into a smaller
    /// cache (e.g. memory cache).
    #[arg(long)]
    num_active_fragments: Option<u32>,

    /// JSON output path for results.
    #[arg(long, default_value = "pe-throughput-results.json")]
    output: String,
}

// ---------------------------------------------------------------------------
// Plan Executor server / client helpers
// ---------------------------------------------------------------------------

struct ServerHandle {
    cancel: CancellationToken,
    join: tokio::task::JoinHandle<()>,
}

impl ServerHandle {
    async fn shutdown(self) {
        self.cancel.cancel();
        self.join.await.unwrap();
    }
}

fn build_cache_config(
    cache_mode: &CacheMode,
    cache_size_mb: u64,
    disk_cache_dir: &Path,
) -> Option<ChainedCachesConfig> {
    let capacity = cache_size_mb * 1024 * 1024;
    let page_size = Some(64 * 1024u64);

    match cache_mode {
        CacheMode::Memory => Some(ChainedCachesConfig::new(vec![
            ObjectStoreCacheConfig::InMemory {
                engine: Some(InMemoryCacheEngine::Moka),
                max_size_in_bytes: Some(capacity),
                fraction_of_sys_memory: None,
                page_size,
                exclude_filters: None,
            },
        ])),
        CacheMode::Disk => Some(ChainedCachesConfig::new(vec![
            ObjectStoreCacheConfig::disk(
                [disk_cache_dir],
                capacity,
                page_size,
                Some(2),
                None,
                None,
                None,
            ),
        ])),
        CacheMode::None => None,
    }
}

async fn start_server(
    uds_path: &Path,
    prom_registry: Registry,
    cache_mode: &CacheMode,
    cache_size_mb: u64,
    disk_cache_dir: &Path,
) -> ServerHandle {
    let uds = UnixListener::bind(uds_path).unwrap();
    let uds_stream = UnixListenerStream::new(uds);

    let cancel = CancellationToken::new();
    let server_cancel = cancel.clone();

    let cache_config = build_cache_config(cache_mode, cache_size_mb, disk_cache_dir);

    // Use ServerParams::build — the same wiring as production.
    // Override the default index/metadata cache sizes which are percentage-based
    // and would consume most of the system memory.
    let params = ServerParams {
        cache: cache_config,
        metrics: None,
        index_cache_size_bytes: Some("256M".to_string()),
        metadata_cache_size_bytes: Some("128M".to_string()),
        ..Default::default()
    };

    let srv = params
        .build(None, prom_registry, Arc::new(DefaultPlanExecutor))
        .await
        .expect("failed to build PE server");

    let join = tokio::spawn(async move {
        Server::builder()
            .add_service(srv)
            .serve_with_incoming_shutdown(uds_stream, server_cancel.cancelled())
            .await
            .unwrap();
    });

    ServerHandle { cancel, join }
}

async fn make_client(
    uds_path: impl AsRef<Path> + Clone + Send + 'static,
) -> RemotePlanWorkerClient<Channel> {
    let channel = Endpoint::try_from("http://notarealname")
        .unwrap()
        .connect_with_connector(service_fn(move |_: Uri| {
            let uds_path = uds_path.clone();
            async move {
                let stream = UnixStream::connect(uds_path).await?;
                Ok::<_, std::io::Error>(TokioIo::new(stream))
            }
        }))
        .await
        .unwrap();
    RemotePlanWorkerClient::new(channel)
}

// ---------------------------------------------------------------------------
// Dataset creation
// ---------------------------------------------------------------------------

fn create_schema(vector_dim: usize) -> Arc<Schema> {
    let mut fields = vec![
        Field::new("id", DataType::UInt64, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                vector_dim as i32,
            ),
            false,
        ),
    ];
    for i in 0..4 {
        fields.push(Field::new(
            &format!("str_payload_{}", i),
            DataType::Utf8,
            false,
        ));
    }
    for i in 0..4 {
        fields.push(Field::new(
            &format!("float_payload_{}", i),
            DataType::Float64,
            false,
        ));
    }
    Arc::new(Schema::new(fields))
}

fn generate_batch(
    schema: Arc<Schema>,
    num_rows: usize,
    vector_dim: usize,
    id_offset: u64,
) -> RecordBatch {
    let mut rng = rand::thread_rng();

    let ids: Vec<u64> = (id_offset..id_offset + num_rows as u64).collect();
    let id_array = UInt64Array::from(ids);

    let vector_values: Vec<f32> = (0..num_rows * vector_dim)
        .map(|_| rng.gen::<f32>())
        .collect();
    let vector_values_array = arrow::array::Float32Array::from(vector_values);
    let vector_array = arrow::array::FixedSizeListArray::new(
        Arc::new(Field::new("item", DataType::Float32, true)),
        vector_dim as i32,
        Arc::new(vector_values_array),
        None,
    );

    let mut str_arrays: Vec<Arc<dyn Array>> = Vec::with_capacity(4);
    for _ in 0..4 {
        let mut builder = StringBuilder::with_capacity(num_rows, num_rows * 16);
        for _ in 0..num_rows {
            let len = rng.gen_range(12..=16);
            let s: String = (&mut rng)
                .sample_iter(&Alphanumeric)
                .take(len)
                .map(char::from)
                .collect();
            builder.append_value(&s);
        }
        str_arrays.push(Arc::new(builder.finish()));
    }

    let mut float_arrays: Vec<Arc<dyn Array>> = Vec::with_capacity(4);
    for _ in 0..4 {
        let values: Vec<f64> = (0..num_rows).map(|_| rng.gen::<f64>()).collect();
        float_arrays.push(Arc::new(Float64Array::from(values)));
    }

    let mut columns: Vec<Arc<dyn Array>> = Vec::with_capacity(10);
    columns.push(Arc::new(id_array));
    columns.push(Arc::new(vector_array));
    columns.extend(str_arrays);
    columns.extend(float_arrays);

    RecordBatch::try_new(schema, columns).expect("failed to create record batch")
}

async fn create_dataset(args: &Args) -> Result<Dataset> {
    let num_fragments = args.num_fragments;
    let rows_per_fragment = args.rows_per_fragment as usize;
    let total_rows = num_fragments as usize * rows_per_fragment;
    let vector_dim = args.vector_dim;
    let schema = create_schema(vector_dim);

    println!(
        "Creating dataset at {} ({} fragments, {} rows/fragment, {} total rows)",
        args.dataset_uri, num_fragments, rows_per_fragment, total_rows,
    );
    println!(
        "  Schema: id (u64), vector (f32x{}, {} bytes/row), 4x str_payload, 4x float_payload",
        vector_dim,
        vector_dim * 4,
    );

    let pb = ProgressBar::new(num_fragments as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("  Writing fragments [{bar:40}] {pos}/{len} ({eta})")
            .unwrap(),
    );

    let batch_size = 8192usize.min(rows_per_fragment);
    let batches_per_fragment = (rows_per_fragment + batch_size - 1) / batch_size;

    let mut dataset: Option<Dataset> = None;

    for frag_idx in 0..num_fragments {
        let id_base = frag_idx as u64 * rows_per_fragment as u64;

        let mut frag_batches: Vec<RecordBatch> = Vec::with_capacity(batches_per_fragment);
        for batch_idx in 0..batches_per_fragment {
            let offset = batch_idx * batch_size;
            let rows_in_batch = batch_size.min(rows_per_fragment - offset);
            frag_batches.push(generate_batch(
                schema.clone(),
                rows_in_batch,
                vector_dim,
                id_base + offset as u64,
            ));
        }

        let reader = arrow::record_batch::RecordBatchIterator::new(
            frag_batches.into_iter().map(Ok),
            schema.clone(),
        );

        let params = WriteParams {
            mode: if frag_idx == 0 {
                WriteMode::Create
            } else {
                WriteMode::Append
            },
            max_rows_per_file: rows_per_fragment,
            data_storage_version: Some(LanceFileVersion::V2_1),
            ..Default::default()
        };

        dataset = Some(
            Dataset::write(reader, &args.dataset_uri, Some(params))
                .await
                .with_context(|| format!("Failed to write fragment {}", frag_idx))?,
        );

        pb.inc(1);
    }

    pb.finish();

    let ds = dataset.expect("no fragments written");
    let count = ds.count_rows(None).await?;
    println!(
        "  Dataset created: {} rows across {} fragments",
        count,
        ds.get_fragments().len(),
    );

    Ok(ds)
}

// ---------------------------------------------------------------------------
// Take workload
// ---------------------------------------------------------------------------

fn generate_random_row_ids(
    take_size: usize,
    num_fragments: u32,
    rows_per_fragment: u32,
) -> Vec<u64> {
    let mut rng = rand::thread_rng();
    (0..take_size)
        .map(|_| {
            let frag = rng.gen_range(0..num_fragments) as u64;
            let row = rng.gen_range(0..rows_per_fragment) as u64;
            (frag << 32) | row
        })
        .collect()
}

async fn run_takes(
    client: &RemotePlanWorkerClient<Channel>,
    table_uri: &str,
    version: u64,
    field_ids: &[u32],
    num_takes: usize,
    take_size: usize,
    num_fragments: u32,
    rows_per_fragment: u32,
    concurrency: usize,
    label: &str,
) -> Vec<Duration> {
    let sem = Arc::new(Semaphore::new(concurrency));

    let pb = ProgressBar::new(num_takes as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(&format!("  {} [{{bar:40}}] {{pos}}/{{len}} ({{eta}})", label))
            .unwrap(),
    );

    let mut handles = Vec::with_capacity(num_takes);

    for _ in 0..num_takes {
        let sem = sem.clone();
        let mut client = client.clone();
        let table_uri = table_uri.to_string();
        let field_ids = field_ids.to_vec();
        let pb = pb.clone();
        let row_ids = generate_random_row_ids(take_size, num_fragments, rows_per_fragment);

        handles.push(tokio::spawn(async move {
            let _permit = sem.acquire().await.unwrap();
            let start = Instant::now();
            let _res = client
                .execute_plan(ExecutePlanRequest {
                    table_uri,
                    version,
                    manifest_e_tag: Default::default(),
                    storage_options: Default::default(),
                    plan: Some(Plan::RandomTake(RandomTake {
                        field_ids,
                        row_ids,
                        ..Default::default()
                    })),
                })
                .await
                .expect("RandomTake failed");
            let elapsed = start.elapsed();
            pb.inc(1);
            elapsed
        }));
    }

    let mut latencies = Vec::with_capacity(num_takes);
    for h in handles {
        latencies.push(h.await.unwrap());
    }

    pb.finish();
    latencies
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct LatencyStats {
    count: usize,
    mean_ms: f64,
    std_ms: f64,
    min_ms: f64,
    max_ms: f64,
    p50_ms: f64,
    p95_ms: f64,
    p99_ms: f64,
}

fn compute_stats(latencies: &[Duration]) -> LatencyStats {
    let mut sorted: Vec<f64> = latencies.iter().map(|d| d.as_secs_f64() * 1000.0).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = sorted.len() as f64;
    let mean = sorted.iter().sum::<f64>() / n;
    let variance = sorted.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;

    LatencyStats {
        count: sorted.len(),
        mean_ms: mean,
        std_ms: variance.sqrt(),
        min_ms: sorted[0],
        max_ms: sorted[sorted.len() - 1],
        p50_ms: sorted[(n * 0.50) as usize],
        p95_ms: sorted[(n * 0.95) as usize],
        p99_ms: sorted[sorted.len().saturating_sub(1).min((n * 0.99) as usize)],
    }
}

// ---------------------------------------------------------------------------
// JSON output
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct BenchmarkOutput {
    benchmark_type: String,
    timestamp: u64,
    config: BenchmarkConfig,
    results: BenchmarkResults,
}

#[derive(Serialize)]
struct BenchmarkConfig {
    dataset_uri: String,
    num_fragments: u32,
    rows_per_fragment: u32,
    total_rows: u64,
    vector_dim: usize,
    take_size: usize,
    num_takes: usize,
    concurrency: usize,
    columns: String,
    cache_mode: String,
}

#[derive(Serialize)]
struct BenchmarkResults {
    wall_clock_secs: f64,
    throughput_takes_per_sec: f64,
    throughput_rows_per_sec: f64,
    total_read_iops: u64,
    iops_per_second: f64,
    total_read_bytes: u64,
    read_bandwidth_mb_per_sec: f64,
    latency: LatencyStats,
    values_ns: Vec<u64>,
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    let args = Args::parse();

    let total_rows = args.num_fragments as u64 * args.rows_per_fragment as u64;
    let active_fragments = args.num_active_fragments.unwrap_or(args.num_fragments);

    println!("Plan Executor Throughput Benchmark");
    println!("{}", "=".repeat(60));
    println!("  dataset_uri:      {}", args.dataset_uri);
    println!("  num_fragments:    {}", args.num_fragments);
    if active_fragments != args.num_fragments {
        println!("  active_fragments: {}", active_fragments);
    }
    println!("  rows_per_fragment:{}", args.rows_per_fragment);
    println!("  total_rows:       {}", total_rows);
    println!(
        "  vector_dim:       {} ({} bytes/vector)",
        args.vector_dim,
        args.vector_dim * 4,
    );
    println!("  take_size:        {}", args.take_size);
    println!("  num_takes:        {}", args.num_takes);
    println!("  concurrency:      {}", args.concurrency);
    println!("  columns:          {:?}", args.columns);
    println!("  cache_mode:       {:?}", args.cache_mode);
    println!("  warmup_takes:     {}", args.warmup_takes);
    println!();

    // Step 1: Create dataset if needed
    println!("{}", "=".repeat(60));
    if args.skip_create {
        println!("Step 1: Opening existing dataset");
    } else {
        println!("Step 1: Creating dataset");
    }
    println!("{}", "=".repeat(60));

    let dataset = if args.skip_create {
        Dataset::open(&args.dataset_uri)
            .await
            .context("Failed to open dataset")?
    } else {
        create_dataset(&args).await?
    };

    let count = dataset.count_rows(None).await?;
    let num_fragments = dataset.get_fragments().len();
    let version = dataset.version().version;
    println!("  Dataset: {} rows, {} fragments, version {}", count, num_fragments, version);

    // Resolve field IDs for the selected columns
    let col_names = args.columns.column_names();
    let field_ids: Vec<u32> = col_names
        .iter()
        .map(|name| {
            dataset
                .schema()
                .field(name)
                .unwrap_or_else(|| panic!("field '{}' not found in schema", name))
                .id as u32
        })
        .collect();
    println!(
        "  Projecting {} columns: {:?} (field_ids: {:?})",
        col_names.len(),
        col_names,
        field_ids,
    );

    // Step 2: Start plan executor server
    println!("\n{}", "=".repeat(60));
    println!("Step 2: Starting plan executor server (cache_mode={:?})", args.cache_mode);
    println!("{}", "=".repeat(60));

    let tmp_dir = TempDir::new().expect("failed to create tempdir");
    let uds_path = tmp_dir.path().join("pe.sock");
    let disk_cache_dir = if let Some(ref p) = args.disk_cache_path {
        let dir = std::path::PathBuf::from(p);
        std::fs::create_dir_all(&dir)?;
        dir
    } else {
        let dir = tmp_dir.path().join("disk_cache");
        std::fs::create_dir_all(&dir)?;
        dir
    };

    let prom_registry = Registry::new();

    // Install the metrics recorder so that metrics::counter!() calls
    // (e.g. file_page_cache_reads_total in page_file.rs) are captured.
    let metrics_handle = metrics_exporter_prometheus::PrometheusBuilder::new()
        .build_recorder();
    let metrics_render_handle = metrics_handle.handle();
    metrics::set_global_recorder(metrics_handle)
        .expect("failed to set global metrics recorder");

    let server = start_server(
        &uds_path,
        prom_registry.clone(),
        &args.cache_mode,
        args.cache_size_mb,
        &disk_cache_dir,
    )
    .await;
    let client = make_client(uds_path.clone()).await;
    println!("  Server started on {}", uds_path.display());

    // Step 3: Warmup or Prewarm
    if args.prewarm {
        println!("\n{}", "=".repeat(60));
        println!("Step 3: Prewarming cache (all fragments, all columns)");
        println!("{}", "=".repeat(60));

        let fragment_ids: Vec<u64> = (0..active_fragments as u64).collect();
        let col_names_str: Vec<String> = args.columns.column_names().iter().map(|s| s.to_string()).collect();
        let prewarm_id = uuid::Uuid::new_v4();

        let mut prewarm_client = client.clone();
        let prewarm_start = std::time::Instant::now();
        prewarm_client
            .execute_plan(ExecutePlanRequest {
                table_uri: args.dataset_uri.clone(),
                version,
                manifest_e_tag: Default::default(),
                storage_options: Default::default(),
                plan: Some(Plan::PreWarmCache(PreWarmCache {
                    fragment_ids,
                    columns: col_names_str,
                    id_hi: prewarm_id.as_u64_pair().0,
                    id_lo: prewarm_id.as_u64_pair().1,
                    db: "benchmark".to_string(),
                    table: "pe-throughput".to_string(),
                    fragment_ranges: vec![],
                    concurrency: Some(16),
                })),
            })
            .await
            .expect("PreWarmCache failed");

        // Poll until prewarm completes (it runs asynchronously in the background)
        println!("  Prewarm submitted, waiting for completion...");
        loop {
            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
            let status = prewarm_client
                .get_page_cache_executor_status(PageCacheExecutorStatusRequest {})
                .await
                .expect("get_page_cache_executor_status failed");
            let response = status.into_inner();
            let our_op = response.prewarm_ops.iter().find(|op| {
                op.id_hi == prewarm_id.as_u64_pair().0
                    && op.id_lo == prewarm_id.as_u64_pair().1
            });
            let disk_used_mb = response.disk_used / (1024 * 1024);
            let disk_cap_mb = response.disk_capacity / (1024 * 1024);
            match our_op {
                Some(op) if op.complete => {
                    println!(
                        "  Prewarm completed in {:.2}s (disk cache: {} / {} MB)",
                        prewarm_start.elapsed().as_secs_f64(),
                        disk_used_mb, disk_cap_mb,
                    );
                    break;
                }
                Some(_) => {
                    println!(
                        "  Prewarming... {:.0}s elapsed (disk cache: {} / {} MB)",
                        prewarm_start.elapsed().as_secs_f64(),
                        disk_used_mb, disk_cap_mb,
                    );
                }
                None => {
                    // Operation not found — might have completed and been cleaned up
                    println!(
                        "  Prewarm operation not found in status, assuming complete ({:.0}s)",
                        prewarm_start.elapsed().as_secs_f64(),
                    );
                    break;
                }
            }
        }
    } else if args.warmup_takes > 0 {
        println!("\n{}", "=".repeat(60));
        println!("Step 3: Warmup ({} takes)", args.warmup_takes);
        println!("{}", "=".repeat(60));
        run_takes(
            &client,
            &args.dataset_uri,
            version,
            &field_ids,
            args.warmup_takes,
            args.take_size,
            active_fragments,
            args.rows_per_fragment,
            args.concurrency,
            "Warmup",
        )
        .await;
    }

    // Record pre-timed-run counter values so we can compute the delta
    let iops_before = get_total_read_iops(&prom_registry);
    let bytes_before = get_total_read_bytes(&prom_registry);
    let pre_metrics = metrics_render_handle.render();
    let pre_entries = get_metrics_counter(&pre_metrics, "memory_page_cache_entries");
    let pre_mem_misses = get_metrics_counter(&pre_metrics, "memory_page_cache_misses_total");
    println!("  (warmup IO: {} reads, {:.2} MB, mem_cache_entries={}, mem_misses={})",
        iops_before, bytes_before as f64 / (1024.0 * 1024.0), pre_entries, pre_mem_misses);

    // Step 4: Timed run
    println!("\n{}", "=".repeat(60));
    println!("Step 4: Timed run ({} takes)", args.num_takes);
    println!("{}", "=".repeat(60));

    let wall_start = Instant::now();
    let latencies = run_takes(
        &client,
        &args.dataset_uri,
        version,
        &field_ids,
        args.num_takes,
        args.take_size,
        active_fragments,
        args.rows_per_fragment,
        args.concurrency,
        "Timed",
    )
    .await;
    let wall_elapsed = wall_start.elapsed();

    // Collect IO metrics from prometheus registry
    let iops_after = get_total_read_iops(&prom_registry);
    let bytes_after = get_total_read_bytes(&prom_registry);
    let total_read_iops = iops_after - iops_before;
    let total_read_bytes = bytes_after - bytes_before;
    let iops_per_second = total_read_iops as f64 / wall_elapsed.as_secs_f64();
    let read_bandwidth_mb_per_sec =
        total_read_bytes as f64 / (1024.0 * 1024.0) / wall_elapsed.as_secs_f64();

    // Step 5: Results
    println!("\n{}", "=".repeat(60));
    println!("BENCHMARK RESULTS");
    println!("{}", "=".repeat(60));

    let stats = compute_stats(&latencies);
    let throughput_takes = args.num_takes as f64 / wall_elapsed.as_secs_f64();
    let throughput_rows = throughput_takes * args.take_size as f64;

    println!("\nWall clock:         {:.2}s", wall_elapsed.as_secs_f64());
    println!("Throughput:         {:.2} takes/sec", throughput_takes);
    println!("                    {:.2} rows/sec", throughput_rows);
    println!();
    println!("Object Store I/O:");
    println!("  Total read IOPS:  {}", total_read_iops);
    println!("  IOPS/second:      {:.2}", iops_per_second);
    println!("  Total read bytes: {:.2} MB", total_read_bytes as f64 / (1024.0 * 1024.0));
    println!("  Read bandwidth:   {:.2} MB/s", read_bandwidth_mb_per_sec);
    println!("  IOPS/take:        {:.1}", total_read_iops as f64 / args.num_takes as f64);

    // Page cache metrics (from the metrics crate, not the prometheus registry)
    let metrics_output = metrics_render_handle.render();
    let page_cache_reads = get_metrics_counter(&metrics_output, "file_page_cache_reads_total");
    let page_cache_races = get_metrics_counter(&metrics_output, "file_page_cache_race_evictions_total");
    let mem_cache_reads = get_metrics_counter(&metrics_output, "memory_page_cache_reads_total");
    let mem_cache_misses = get_metrics_counter(&metrics_output, "memory_page_cache_misses_total");
    if page_cache_reads > 0 || page_cache_races > 0 || mem_cache_reads > 0 || mem_cache_misses > 0 {
        println!();
        println!("Page Cache:");
        if page_cache_reads > 0 {
            println!("  Disk cache reads: {}", page_cache_reads);
        }
        if page_cache_races > 0 {
            println!("  Race evictions:   {}", page_cache_races);
        }
        if mem_cache_reads > 0 || mem_cache_misses > 0 {
            println!("  Mem cache hits:   {}", mem_cache_reads);
            println!("  Mem cache misses: {}", mem_cache_misses);
        }
    }

    println!("\nLatency (ms):");
    println!("  Mean:  {:.3}", stats.mean_ms);
    println!("  Std:   {:.3}", stats.std_ms);
    println!("  Min:   {:.3}", stats.min_ms);
    println!("  Max:   {:.3}", stats.max_ms);
    println!("  p50:   {:.3}", stats.p50_ms);
    println!("  p95:   {:.3}", stats.p95_ms);
    println!("  p99:   {:.3}", stats.p99_ms);

    // Write JSON output
    let values_ns: Vec<u64> = latencies.iter().map(|d| d.as_nanos() as u64).collect();

    let output = BenchmarkOutput {
        benchmark_type: "pe_throughput".to_string(),
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs(),
        config: BenchmarkConfig {
            dataset_uri: args.dataset_uri.clone(),
            num_fragments: args.num_fragments,
            rows_per_fragment: args.rows_per_fragment,
            total_rows,
            vector_dim: args.vector_dim,
            take_size: args.take_size,
            num_takes: args.num_takes,
            concurrency: args.concurrency,
            columns: format!("{:?}", args.columns),
            cache_mode: format!("{:?}", args.cache_mode),
        },
        results: BenchmarkResults {
            wall_clock_secs: wall_elapsed.as_secs_f64(),
            throughput_takes_per_sec: throughput_takes,
            throughput_rows_per_sec: throughput_rows,
            total_read_iops,
            iops_per_second,
            total_read_bytes,
            read_bandwidth_mb_per_sec,
            latency: stats,
            values_ns: values_ns.clone(),
        },
    };

    if let Some(parent) = std::path::Path::new(&args.output).parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&args.output, serde_json::to_string_pretty(&output)?)?;
    println!("\nResults written to {}", args.output);

    println!("\n{}", "=".repeat(60));
    println!("Shutting down server...");
    server.shutdown().await;
    println!("Benchmark Complete!");
    println!("{}", "=".repeat(60));

    Ok(())
}
