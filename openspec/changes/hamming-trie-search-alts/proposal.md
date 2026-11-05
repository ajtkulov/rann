## Why

The current Bloom-trie ANN path uses A* with heuristic `h = 0` (Dijkstra on Hamming) over `NN`-bit edges. On large indexes this expansion can hit `max_expansions` before collecting good candidates, and the design doc (§5) only commits to A*. We need a clear menu of alternative min-Hamming (or near-min) trie search strategies so we can pick, implement, and benchmark approaches beyond A*—especially ones that bound work better under Bloom false positives.

## What Changes

- Document and select alternative search strategies for traversing the dual-Bloom virtual trie to find low Hamming-distance hash codes relative to a query code.
- Add a pluggable `HammingSearchStrategy` (or equivalent) in the library so A* is not the only path.
- Implement at least two non-A* strategies suitable for the Bloom+`NN`-chunk trie (see design), plus keep A* as baseline.
- Extend the existing benchmark harness to compare strategies on the same loaded dataset (candidate count, expansions, latency, recall vs exact fullscan).
- Update specs for ANN search to require strategy selection without changing the dual-Bloom index format.

## Capabilities

### New Capabilities
- `hamming-search-strategies`: Pluggable algorithms for low-Hamming candidate discovery on the Bloom-gated trie (beyond A*/Dijkstra).
- `search-strategy-bench`: Benchmark and compare Hamming search strategies on a loaded dataset against exact fullscan ground truth.

### Modified Capabilities

- (none yet — `ann-knn-search` lives in the unarchived `rust-ann-knn-bloom-trie` change; strategy requirements are captured under new capability `hamming-search-strategies` and can be synced into main specs on archive)

## Impact

- Touches `src/search.rs` (and likely a new `src/search_strategies/` module); public `TrieIndex::search` gains an optional strategy / builder param.
- Benchmark binary gains `--strategy` (or multi-strategy compare mode).
- No change to `.rann` on-disk format; Bloom insert layout unchanged.
- Recall/latency tradeoffs change by strategy; document when exact min-Hamming is not guaranteed (beam, multi-probe, etc.).
