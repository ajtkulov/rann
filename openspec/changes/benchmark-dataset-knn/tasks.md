## 1. Dependencies and shared types

- [x] 1.1 Add `serde` / `serde_json` (and keep using existing `rand`/`rand_chacha`) to `Cargo.toml`
- [x] 1.2 Define benchmark result structs matching JSON schema v1 (`RunReport`, query cohorts, per-K timings, summary aggregates)

## 2. Benchmark runner CLI

- [x] 2.1 Create `src/bin/benchmark.rs` with args: dataset prefix, `--seed`, `--mode exact|ann`, `-o` stem, optional `--k` list
- [x] 2.2 Load `.rann` (+ `.vocab` if present) fully into memory; record `load_ms`
- [x] 2.3 Build query set: first 20 ids + 20 seeded random ids (no overlap when `n > 40`)
- [x] 2.4 For each query × each K in `{3,5,10,20,50}`, run exact (default) or ANN search; record `elapsed_ms` and neighbors
- [x] 2.5 Compute `summary.by_k` and `summary.by_cohort_k` (mean, min, max, p50)

## 3. JSON report

- [x] 3.1 Serialize the full run report to `<stem>.json`
- [x] 3.2 Unit/integration test on a tiny synthetic index: JSON parses, cohorts sized correctly, K grid present

## 4. HTML visualization

- [x] 4.1 Generate offline-friendly `<stem>.html` with inline SVG (or CSS) latency-vs-K bars for overall + first/random cohorts and a summary table
- [x] 4.2 Wire HTML emission into the CLI by default alongside JSON

## 5. Docs and canvas presentation

- [x] 5.1 Document `benchmark` usage in README (example against `data/glove.6B.300d`)
- [x] 5.2 After a real/sample run, present results via Cursor canvas reading the JSON (latency-by-K + table)
