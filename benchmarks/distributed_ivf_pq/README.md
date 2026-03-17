# Distributed IVF_PQ Benchmark

This benchmark measures the local end-to-end flow of Lance distributed `IVF_PQ`
index building:

1. Create a dataset with multiple fragments.
2. Train shared IVF/PQ parameters once.
3. Split fragments into shard groups.
4. Build shard-local partial indices concurrently.
5. Finalize the distributed merge.
6. Commit the merged index.

The benchmark prints a JSON result with timing fields such as
`train_shared_params_ms`, `shard_build_ms`, and `finalize_ms`.

## Prerequisite

This crate depends on a local Lance checkout at `lance-bench/lance/`. That
matches the existing GitHub Actions convention in this repository.

Example:

```bash
git clone https://github.com/lancedb/lance-bench.git
cd lance-bench
git clone https://github.com/lance-format/lance.git lance
```

## Run

```bash
cargo run --manifest-path benchmarks/distributed_ivf_pq/Cargo.toml --release -- \
  --fragments 8 \
  --rows-per-fragment 262144 \
  --shards 8 \
  --dim 128 \
  --num-partitions 32768 \
  --num-sub-vectors 16 \
  --max-iters 10 \
  --sample-rate 8 \
  --cleanup
```

## Output

The benchmark emits a single JSON object. Important fields:

- `dataset_prepare_ms`
- `train_shared_params_ms`
- `shard_build_ms`
- `finalize_ms`
- `commit_ms`
- `total_index_build_ms`
- `total_pipeline_ms`

For large runs, you may need to increase `RLIMIT_NOFILE`.
