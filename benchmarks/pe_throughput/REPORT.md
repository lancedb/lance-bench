# Plan Executor Take Throughput Benchmark Report

## Overview

This benchmark measures the throughput of random take operations through a real plan executor (PE) gRPC server against a Lance v2.1 dataset stored in S3. The goal is to understand how different caching strategies affect take performance and to identify bottlenecks.

### Setup

- **Instance**: EC2 with 62GB RAM, EBS gp3 volume (6,000 provisioned IOPS)
- **Dataset**: 1,024 fragments, 16,384 rows/fragment (16.7M total rows), stored at `s3://weston-s3-lance-test/pe-throughput`
- **Row schema**: id (u64), vector (f32x1024 = 4KiB), 4x string columns (12-16 chars), 4x float64 columns
- **File format**: Lance v2.1, one file per fragment (~66MB each, ~66GB total)
- **Benchmark**: Issues concurrent `RandomTake` gRPC requests over Unix domain socket to a PE server. Each take fetches K random row addresses across all 10 columns.
- **Concurrency**: 16 concurrent takes in flight

### Cache modes tested

- **None**: Every read hits S3 directly
- **Disk**: Read-through disk page cache (O_DIRECT, 64KB pages) backed by EBS gp3
- **Memory**: In-memory Moka page cache (64KB pages)

For disk and memory cache tests, the cache was prewarmed using the PE's `PreWarmCache` gRPC endpoint before the timed run.

## Results

All tests use 250 active fragments (~16GB working set), all 10 columns projected, concurrency=16.

### Throughput comparison

| Take size | No cache | Disk cache | Memory cache |
|-----------|----------|------------|--------------|
| 1 row | 92.6 takes/s | 740.6 takes/s | **9,758 takes/s** |
| 10 rows | 39.7 takes/s | 59.3 takes/s | **3,374 takes/s** |
| 100 rows | 5.5 takes/s | 6.1 takes/s | **551 takes/s** |

### Latency comparison (mean)

| Take size | No cache | Disk cache | Memory cache |
|-----------|----------|------------|--------------|
| 1 row | 171 ms | 21 ms | **1.6 ms** |
| 10 rows | 397 ms | 269 ms | **4.6 ms** |
| 100 rows | 2,896 ms | 2,587 ms | **29 ms** |

### Speedup vs no cache

| Take size | Disk cache | Memory cache |
|-----------|------------|--------------|
| 1 row | 8x | **105x** |
| 10 rows | 1.5x | **85x** |
| 100 rows | 1.1x | **100x** |

## Analysis

### IOPS per take

Each take of K rows across 10 columns requires approximately 14 object store reads per row (1 per non-string column, 2 per string column for offsets + data). With the page cache, these become 64KB page reads.

| Take size | Disk cache reads/take | IOPS rate |
|-----------|-----------------------|-----------|
| 1 row | 13.4 | ~10,000/s |
| 10 rows | 112.2 | ~6,700/s |
| 100 rows | 1,039.1 | ~6,400/s |

### Disk cache is EBS IOPS-limited

The EBS gp3 volume is provisioned for 6,000 IOPS. At take_size=100, the disk cache saturates at ~6,200 IOPS/s, providing only marginal improvement over S3. The disk cache shines at take_size=1 where each take needs only ~13 pages and the per-request S3 latency (~170ms) dominates.

### Memory cache eliminates I/O

The memory cache delivers ~100x improvement across all take sizes by eliminating all I/O. Latency drops from hundreds of milliseconds (S3 round trips) to single-digit milliseconds (in-memory hash lookup + memcpy). The memory cache is CPU-bound, not I/O-bound.

### Prewarm behavior

- **Disk cache prewarm**: ~75 minutes to fill 66GB at ~15 MB/s (S3 download rate)
- **Memory cache prewarm**: ~60 seconds for 250 fragments (16GB), limited by S3 bandwidth and concurrent fragment scanning
- Prewarm uses the PE's `PreWarmCache` gRPC endpoint which scans all fragments via `FilteredReadExec`
- Prewarm runs asynchronously; the benchmark polls `get_page_cache_executor_status` until completion

## Bugs found

### 1. Disk page cache: partial page corruption (fixed)

**PR**: lancedb/sophon#5590

The `actual_len` calculation in `FilePageCache::get_range_in_page` mixed page-local coordinates with cache-file coordinates:

```rust
// BUGGY
let actual_len = min(range.end, page_info.length + page_offset) - range.start;
// FIXED
let actual_len = min(range.end, page_info.length) - range.start;
```

When reading partial pages (last page of a file, shorter than `page_size`) with `page_id > 0`, the buggy formula never clamped to the stored data length, returning stale bytes from the cache file. This caused lance's decoder to panic on corrupted data.

**Manifestation**: `the offset + length of the sliced Buffer cannot exceed the existing length` panic in lance-encoding at `buffer.rs:290`.

**Unit test**: `test_get_range_partial_page` — pre-fills page slots to force non-zero page_id, verifies partial page reads are clamped correctly.

### 2. Disk page cache: TOCTOU race on slot reuse (fixed)

**PR**: lancedb/sophon#5595

When the disk page cache is full, evicted page slots are reused. A concurrent reader that looked up a `PageInfo` before eviction could read from a slot that had since been overwritten with different data.

**Fix**: Post-read validation — after `read_exact_at`, re-check `page_map.get()` to verify the page is still mapped to the same slot. If evicted, return `None` (cache miss) instead of corrupted data. The caller fetches from S3.

**Manifestation**: Same `buffer.rs:290` panic as the partial page bug, but only under concurrent load (concurrency >= 2) with a cache smaller than the working set.

**Unit test**: `test_concurrent_read_write_eviction_race` — uses `#[cfg(test)] yield_now()` between `page_map.get()` and `read_exact_at` to reliably reproduce the race. Detects ~200+ corrupted reads without the fix.

### 3. ServerParams::default() memory allocation

`ServerParams::default()` allocates 65% of system memory for the page cache, 20% for the index cache, and 10% for the metadata cache (95% total). On a 62GB machine, this easily causes OOM. The benchmark overrides `index_cache_size_bytes` and `metadata_cache_size_bytes` to `256M` and `128M` respectively.

### 4. InstrumentedObjectStoreWrapper per-request counter isolation

Each call to `InstrumentedObjectStoreWrapper::wrap()` creates a new `InstrumentedObjectStore` with fresh prometheus counter objects. Only the first instance's counters are registered in the `Registry`; subsequent instances' counters are orphaned (prometheus returns `AlreadyReg`). This means per-request wrapping in `wrap_dataset_for_request` produces counters that don't accumulate in the registry.

In production (phalanx), this is not a problem because the `InstrumentedObjectStoreWrapper` is applied once at dataset-open time via `FsCatalogBuilder::object_store_wrapper`, not per-request.

The benchmark works around this by using the `metrics` crate counters (`file_page_cache_reads_total`, `memory_page_cache_reads_total`) which go through a global recorder, instead of the per-request prometheus counters.

## Benchmark implementation notes

### Architecture

The benchmark spawns a real PE gRPC server over a Unix domain socket using `ServerParams::build`. It sends `RandomTake` requests via `RemotePlanWorkerClient`. This matches the production request path: gRPC → `RemotePlanWorkerImpl::execute_plan` → `wrap_dataset_for_request` → `execute_take_rows`.

### Row address encoding

Row addresses encode `(fragment_id << 32) | row_offset`. The benchmark generates random addresses within the configured fragment range. The `--num-active-fragments` flag limits both the prewarm and takes to the first N fragments, enabling the working set to fit in smaller caches (e.g., memory).

### Metrics

- **Object store IOPS**: `object_store_*_calls_total` from prometheus Registry (only works when `InstrumentedObjectStoreWrapper` is applied at dataset-open time, not via `ServerParams::build`)
- **Disk cache reads**: `file_page_cache_reads_total` from the `metrics` crate
- **Memory cache hits/misses**: `memory_page_cache_reads_total` / `memory_page_cache_misses_total` from the `metrics` crate
- **Race evictions**: `file_page_cache_race_evictions_total` from the `metrics` crate

### Key files

- `benchmarks/pe_throughput/src/main.rs` — main benchmark binary
- `benchmarks/pe_throughput/src/bin/trace_reads.rs` — standalone I/O tracing tool
- `benchmarks/pe_throughput/src/bin/direct_take.rs` — direct lance take test (bypasses PE)
- `sophon-caching/src/object_store/page_file.rs` — disk page cache (bugs fixed here)
- `sophon-caching/src/object_store/moka.rs` — memory page cache
- `sophon-caching/src/object_store/paging.rs` — read-through page cache logic
- `sophon-caching/AGENTS.md` — architecture documentation for the caching layer
