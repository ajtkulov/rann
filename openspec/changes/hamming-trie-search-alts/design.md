## Context

Today `TrieIndex::collect_candidates` expands a Bloom-gated radix trie with A* where `f = g` (Hamming on decided hash bits; `h = 0`). Edges are `NN`-bit labels (default 64-way). Bloom false positives inflate the fringe; `max_expansions` truncates search. Design §5 of `rust-ann-knn-bloom-trie` only specified A*.

Goal: catalog stronger / cheaper alternatives for **finding low Hamming-distance hash paths** (then keep existing id-suffix enumeration + cosine rerank), pick implementable ones, and make search strategy pluggable.

## Goals / Non-Goals

**Goals:**
- Enumerate credible alternatives to A* for min / near-min Hamming search on this trie shape.
- Choose 2+ implementable strategies besides A*, with clear exactness guarantees.
- Pluggable API; same Bloom index; comparable via benchmark vs fullscan ground truth.
- Prefer strategies that reduce expansions under FP noise or bound latency.

**Non-Goals:**
- Replacing LSH / Bloom layout.
- Learning-to-hash or neural indexers.
- Guaranteeing exact cosine ANN (still approximate via LSH + optional incomplete Hamming search).

## Problem shape (constraints any algorithm must respect)

1. **Implicit trie:** no material children lists — existence of an edge is a Prefix Bloom probe on `prefix || label`.
2. **Branching factor `B = 2^NN`** (64 at NN=6) per depth; depth `L = ceil(H/NN)` (43 for H=256).
3. **Cost:** Hamming distance is additive over bits/chunks vs query code `q`.
4. **False positives:** Bloom may admit edges that lead nowhere real; Terminal BF + sidecar filter phantoms later.
5. **Objective for hash phase:** surface hash codes with small `d_H(code, q)`, ideally in increasing order until budget / enough terminals.

## Alternative approaches (beyond A*)

### A. Kept baseline: A* / Dijkstra (`h = 0`)
- Current behavior. Exact min-Hamming order among explored Bloom-positive paths until budget.
- **Pros:** simple, admissible. **Cons:** large fringe; FPs waste expansions; weak under budget.

### B. A* with non-trivial heuristic (`h > 0`)
- Still A*, but sharper: e.g. `h = 0` on remaining bits if unknown; or layer-wise lower bound using mandatory mismatches if some bits of `q` have no Bloom-positive sibling (hard without negative certificates).
- Practical variant: **chunk-wise lower bound** — at depth `d`, for each future chunk, if we could prove no label achieves Hamming 0 against `q`'s chunk, add 1… — Bloom cannot prove absence reliably (no FNs on inserted keys, but cannot trust negatives for “no edge”).
- **Verdict:** weak gains on Bloom tries; optional micro-optimization only. Not a primary “beyond A*” pick.

### C. Multi-probe / increasing-radius search (LSH classic)
- Start with radius `r = 0`: only follow labels that match `q`'s chunk exactly (Hamming 0 per chunk).
- If too few candidates, allow radius `r = 1, 2, …` up to `r_max` (or until K terminals).
- At each radius, BFS/DFS only edges whose **chunk Hamming ≤ remaining budget**.
- **Pros:** natural for LSH; work grows with radius; easy to stop early. **Cons:** not a single priority queue; may revisit prefixes when radius increases (cache visited `(depth, prefix, r)`).
- **Exactness:** finds all codes with `d_H ≤ r` that Bloom admits before moving on → exact for bounded radius.
- **Primary candidate.**

### D. Beam search (width-`W` best-first)
- At each depth keep only the `W` prefixes with smallest `g` (Hamming so far), expand those only.
- **Pros:** hard latency/memory bound `O(W · L · B)` probes. **Cons:** can drop the globally optimal path (not exact min-Hamming).
- Tunable recall vs speed; good default for production latency SLOs.
- **Primary candidate.**

### E. Best-first with bucketed Dijkstra (Dial / radix heap)
- Hamming `g` is a small integer `0..H`. Use **Dial’s algorithm**: array of deques indexed by `g`, pop from lowest non-empty bucket.
- Same exploration order as Dijkstra/`h=0` A*, but cheaper than binary heap for integer costs.
- **Pros:** faster priority queue, same exactness as current A*. **Cons:** does not reduce expansions under FPs — only constant-factor PQ cost.
- **Secondary candidate** (optimize A*, not a new search philosophy).

### F. Layer-wise dynamic programming / trellis
- Treat chunks as time steps; state = (depth, hamming_so_far) or (depth, last few bits).
- Propagate reachable Bloom-positive prefixes; prune dominated states (same depth + prefix bits… prefixes are many).
- Full DP over all prefixes is exponential; practical form collapses to **beam** (D) or **radius layers** (C).
- **Verdict:** subsumed by C/D; document as unifying view, don’t implement separately.

### G. Random / MCMC walk on the trie
- From root, sample labels biased toward low chunk Hamming among Bloom-positive edges; restart.
- **Pros:** anytime. **Cons:** high variance, weak guarantees; skip for v1.

### H. Explicit invert: generate query neighborhoods offline
- Enumerate all bitstrings within Hamming `r` of `q` (cost `∑_i C(H,i)`), probe Terminal/Prefix BF for each code (no trie walk).
- For H=256, even `r=2` is huge (`C(256,2)≈32k` codes; `r=3` ~ 2.7M; `r≥4` painful).
- With chunk structure, generate only codes that survive Prefix BF chunk-by-chunk (becomes C).
- **Useful for tiny `r_max`** as a special case of multi-probe.

### I. Bidirectional search
- Forward from query-guided root; backward needs terminal seeds — we don’t have inverted lists of ids by hash.
- **Verdict:** not applicable without inverted posting lists.

### J. Guided bit-flip local search at leaves
- Greedy: take exact-match path if Bloom allows; then flip bits of the hash one-by-one and re-probe Terminal BF / re-walk.
- **Pros:** very fast when near-duplicates exist. **Cons:** incomplete; good as a **bootstrap** before wider multi-probe.

## Decisions

### 1. Strategy trait
```text
HammingSearchStrategy::discover_hash_paths(index, query_hash, limits) -> Vec<(BitCode /*H bits*/, g)>
```
then shared `enumerate_ids` + cosine rerank (unchanged).

### 2. Implement in this change
| ID | Strategy | Exact min-Hamming? | Why |
|----|----------|--------------------|-----|
| `astar` | Current A*/Dijkstra | Among explored, until budget | Baseline |
| `multiprobe` | Increasing radius (C) | For all `d_H ≤ r` before bumping `r` | LSH-native, early exit |
| `beam` | Beam width `W` (D) | No | Bounded expansions |

Optional stretch: `dial` bucket queue behind `astar` (E) as an implementation detail.

### 3. Defaults
- Library default remains `astar` for compatibility.
- Benchmark default for ANN compare: run `astar`, `multiprobe`, `beam` with documented params (`r_max`, `W`, `max_expansions`).

### 4. Multi-probe parameters
- `r_max` default: e.g. 8 (tunable); stop early when `min_candidates` terminals collected.
- Within a radius, expand depth-first or BFS with chunk constraint `hamming(label, q_chunk) ≤ r_remaining`.

### 5. Beam parameters
- `beam_width` default: e.g. 32 or 64; at each depth sort survivors by `g`, keep top `W`.

### 6. Evaluation
- Same GloVe DS, same queries as existing bench; metrics: latency, expansions/probes, candidate count, recall@K vs fullscan ground truth.

## Risks / Trade-offs

- [Beam misses true min-Hamming codes] → Mitigate with larger `W` or hybrid: beam then multiprobe refill.
- [Multi-probe explosion at large `r`] → Cap `r_max` and Prefix BF gating; measure on GloVe.
- [Strategy API churn] → Keep `search()` default; add `search_with_strategy(...)`.
- [Fairness vs A*] → Shared expansion counter / probe metrics in reports.

## Migration Plan

- Additive API; existing indexes unchanged.
- Benchmark JSON may add `strategy` field (already has `mode`); extend schema carefully (version bump if needed).

## Open Questions

- Default `r_max` / `beam_width` after first GloVe numbers.
- Whether `multiprobe` should replace `astar` as the recommended ANN default once recall/latency wins.
