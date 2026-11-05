## 1. Strategy API skeleton

- [x] 1.1 Extract shared hash-path → id enumeration + rerank helpers from current A* `collect_candidates`
- [x] 1.2 Introduce `HammingSearchStrategy` enum/trait (`astar`, `multiprobe`, `beam`) and `search_with_strategy` API; keep `search` defaulting to `astar`
- [x] 1.3 Add strategy params struct: `max_expansions`, `r_max`, `min_candidates`, `beam_width`

## 2. Implement alternatives

- [x] 2.1 Refactor existing A* into the `astar` strategy (behavior-preserving)
- [x] 2.2 Implement `multiprobe` increasing-radius search with Prefix BF gating and early stop
- [x] 2.3 Implement `beam` search with configurable `beam_width`
- [x] 2.4 Unit tests: identical query finds self under all strategies on a tiny index; beam respects width; multiprobe explores r=0 before r=1

## 3. Benchmark comparison

- [x] 3.1 Extend `benchmark` CLI with `--strategy` and/or `--strategies astar,multiprobe,beam` for ANN mode
- [x] 3.2 Record per-strategy latency, expansions/probes, candidate counts, and recall@K vs fullscan ground truth in JSON
- [x] 3.3 Run on `data/glove.6B.300d` and write `data/benchmarks/glove-strategies.*`

## 4. Docs

- [x] 4.1 Document strategies and parameters in README (when to use multiprobe vs beam vs astar)
- [x] 4.2 Note in design/README that Dial bucket-queue and bit-flip bootstrap are deferred
