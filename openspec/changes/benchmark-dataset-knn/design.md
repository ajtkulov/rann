## Context

`rann` already serializes GloVe-scale indexes (`.rann` + `.vocab`) and supports exact cosine scan and ANN search. Operators need a fixed, reproducible microbenchmark: load one dataset into memory, query a mixed set (first 20 + 20 random), sweep `K`, time each search, dump JSON, and visualize latency.

Default target dataset: `data/glove.6B.300d` (400k × 300d). Existing bins: `gen_ds`, `query_ds`.

## Goals / Non-Goals

**Goals:**
- One CLI entrypoint that loads a DS fully into RAM and runs the specified query protocol.
- Query set = first 20 ids (0..19) + 20 distinct random ids (seeded; exclude duplicates with the first-20 set when possible).
- For each query × each `K ∈ {3,5,10,20,50}`, run KNN, record elapsed ms and neighbor list.
- Write a single JSON results file with run metadata and aggregates (mean/p50/p95/min/max per K and per cohort: `first` vs `random`).
- Emit a self-contained HTML visual report (latency vs K charts + summary tables) from that JSON.
- Support presenting the same JSON in a Cursor canvas when reviewing a run.

**Non-Goals:**
- Continuous CI performance gates / statistical significance testing.
- Multi-threaded throughput benches (rayon) in v1 — single-threaded wall times only.
- Changing the index format or core search algorithms.
- Distributed / multi-dataset comparison UI beyond one run file.

## Decisions

### 1. CLI binary `benchmark`
- `cargo run --release --bin benchmark -- <prefix> [options]`
- Options: `--seed` (default `1729`), `--mode exact|ann` (default `exact` for trustworthy neighbors + stable timings), `-o` output stem (default `data/benchmarks/<dataset>-<timestamp>`), `--k` override list optional.
- **Alternative:** fold into `query_ds` — rejected; keep query UX separate from batch bench.

### 2. Search mode default = exact
- Exact cosine over in-memory vectors matches prior `query_ds` reliability on GloVe and avoids ANN expansion-budget noise dominating timings.
- `--mode ann` still available for comparing Bloom-trie A* latency/recall later.
- **Alternative:** ANN-only — rejected for v1 primary numbers.

### 3. Timing methodology
- Use `std::time::Instant` around each individual `search` / exact top-K call (not including index load).
- Record `load_ms` separately in metadata.
- No warm-up loop required for v1; document that first query may include allocator noise. Optional `--warmup 1` flag if cheap.
- Aggregates computed in-process before JSON write.

### 4. JSON schema (version 1)
```json
{
  "version": 1,
  "dataset": "data/glove.6B.300d",
  "mode": "exact",
  "seed": 1729,
  "k_values": [3,5,10,20,50],
  "load_ms": 700.0,
  "num_vectors": 400000,
  "dim": 300,
  "queries": [
    {
      "cohort": "first",
      "id": 0,
      "word": "the",
      "runs": [
        { "k": 10, "elapsed_ms": 12.3, "neighbors": [{"id": 0, "word": "the", "distance": 0.0}] }
      ]
    }
  ],
  "summary": {
    "by_k": {
      "10": { "n": 40, "mean_ms": 11.0, "p50_ms": 10.8, "p95_ms": 13.0, "min_ms": 9.5, "max_ms": 14.2 }
    },
    "by_cohort_k": { "first": { "10": { "...": "..." } }, "random": { "10": { "...": "..." } } }
  }
}
```
- Use `serde_json` for serialization (add dependency).

### 5. Visualization
- **HTML report**: binary (or same CLI with `--html`) writes `*.html` embedding Chart.js via CDN or pure SVG/CSS bars — prefer **no network dependency**: inline SVG bar charts + HTML tables generated from the JSON so the file opens offline.
- **Canvas**: a `benchmark-results.canvas.tsx` (or run-specific path under project) that imports/parses the latest JSON (or accepts pasted path) to show latency-by-K and cohort comparison. Created when presenting results after a run (apply phase / opsx follow-up), not required to block the CLI.
- CLI always writes JSON; HTML written alongside by default.

### 6. Random cohort sampling
- Seeded `ChaCha8Rng` / `StdRng` with `--seed`.
- Sample 20 unique ids uniformly from `[0, n)` excluding `{0..19}` when `n > 40`; if `n ≤ 40`, sample without replacement from remaining ids (or allow overlap only if unavoidable and flag in JSON).

## Risks / Trade-offs

- [Large JSON with full neighbor lists at K=50 × 40 queries] → Cap neighbor payload to requested K; still ~2k neighbor records — acceptable. Offer `--no-neighbors` to store timings only if needed.
- [ANN mode slow/incomplete on 400k] → Default exact; document ANN as experimental for this harness.
- [CDN charts fail offline] → Inline SVG, no CDN.
- [Timing variance on laptops] → Report distribution stats; single-run caveat in HTML header.

## Migration Plan

- N/A for library. New binary + output directory `data/benchmarks/` (gitignore optional large HTML/JSON if desired; keep samples).

## Open Questions

- Whether to include ANN-vs-exact recall@K in the same JSON in a follow-up.
- Whether canvas should live in-repo as a template or only be generated per conversation.
