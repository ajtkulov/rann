## Why

We can build and query serialized GloVe datasets, but have no repeatable way to measure KNN latency across query mixes and `K` values, or to share those results as structured data and charts. A benchmark harness is needed now to characterize search cost on real datasets (e.g. `data/glove.6B.300d`) and to compare exact vs ANN paths over time.

## What Changes

- Add a `benchmark` CLI (or extend tooling) that loads a serialized dataset (`.rann` + optional `.vocab`) fully into memory.
- Select **20 first** vectors/words and **20 random** vectors as queries (seeded RNG for reproducibility).
- For each query, run top-K KNN for `K ∈ {3, 5, 10, 20, 50}` and record wall-clock timings (per query, per K, and aggregates).
- Save a machine-readable **JSON** results file (metadata, query set, per-query neighbors + timings, summaries).
- Present results in a **visual** form: a self-contained HTML timing chart/table report generated from the JSON, and a Cursor canvas view for interactive inspection when reviewing runs in-chat.

## Capabilities

### New Capabilities
- `knn-benchmark-runner`: Load an in-memory dataset index; build the fixed+random query set; run timed top-K KNN for the specified K grid.
- `benchmark-json-report`: Persist benchmark runs as versioned JSON (params, timings, neighbor ids/words, aggregates).
- `benchmark-visualization`: Render benchmark JSON as charts/tables (HTML report artifact + canvas presentation).

### Modified Capabilities

- (none — additive tooling on top of existing index load/search)

## Impact

- New binary under `src/bin/` (e.g. `benchmark`) and optional small report generator.
- Depends on existing `TrieIndex::load` / search (exact and/or ANN); reads `data/*.rann` (+ `.vocab` for word labels).
- Outputs under `data/benchmarks/` (JSON + HTML); may add a `.canvas.tsx` for Cursor-side visualization of the latest run.
- No change to on-disk index format; no **BREAKING** API changes to the library.
