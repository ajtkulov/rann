## Context

Greenfield Rust crate (`rann`) for approximate nearest-neighbor search. Vectors are multi-dimensional floats. The index never materializes an explicit trie of nodes; instead dual Bloom filters encode “prefix exists” and “full hash+id exists,” so traversal is guided by membership queries over `NN`-bit edge labels.

Default hyperparameters from the product brief: `H=132` hash bits, `NN=6` bits per trie edge, random-plane seed `1729`, item id `i32` or `i64`, final ranking by cosine distance.

## Goals / Non-Goals

**Goals:**
- Deterministic LSH: same seed + dims → same hyperplanes → same binary codes.
- Compact in-memory dual-Bloom virtual trie for insert and membership.
- A* search that expands `NN`-bit branches by Hamming cost against the query code, pruning via the prefix Bloom filter.
- Collect candidate ids from terminal Bloom checks, then return top-K sorted by cosine (pluggable distance).
- Persist/reload Bloom bitsets and projection metadata for a dataset.
- Solid unit/integration test coverage for the above.

**Non-Goals:**
- GPU / distributed indexing.
- Exact NN guarantees or theoretical recall SLAs beyond Bloom false-positive caveats.
- Dynamic deletion (Bloom filters do not support reliable delete in v1).
- Network RPC server; library API only for v1.
- Training learned hash functions (random hyperplanes only).

## Decisions

### 1. Crate layout
- Single library crate `rann` with modules: `projection`, `bloom`, `trie_index`, `search`, `distance`, `serde_index`.
- **Alternative considered:** workspace with many crates — deferred; one crate keeps the first version simple.

### 2. Binary code representation
- Store the `H`-bit hash as a packed bit buffer (`BitVec` or `[u64; ceil(H/64)]`). Default `H=132` → 3×`u64` with 60 unused high bits, or a smallvec of bytes.
- Item id appended as little-endian bits after the hash for the terminal key: `H + 32` or `H + 64` bits depending on `IdWidth`.
- **Rationale:** matches the “132 + 32/64 bits still binary” requirement; keeps Bloom keys contiguous.

### 3. Random hyperplanes
- Seed RNG with `1729` (configurable). Draw `H` unit-ish normal vectors in `D` dimensions (same `D` as the dataset).
- Bit `i` = `1` iff `dot(x, plane_i) >= 0`, else `0`.
- Planes stored with the index so search uses the identical projection.
- **Alternative:** angular LSH with more sophisticated sampling — not needed for v1.

### 4. Dual Bloom filters as virtual trie
- **Prefix BF:** for each inserted item, for every prefix length `k * NN` bits of the hash-only portion (`k = 1..ceil(H/NN)`), insert that prefix bitstring. Optionally also insert prefixes of `hash||id` after depth `H` for id-bit traversal.
- **Terminal BF:** insert the full `hash_bits || id_bits` key.
- Chunk size `NN=6` ⇒ up to `2^6 = 64` candidate edges per depth; A* probes each candidate chunk against Prefix BF.
- After depth covering all `H` hash bits, continue expanding over id-bit chunks (still `NN`-aligned, with padding on the last chunk) filtered by Prefix BF, confirming complete keys in Terminal BF.
- **Alternative:** explicit radix trie — higher memory; rejected for the Bloom-backed design goal.

### 5. A* search
- State: `(depth_chunks, prefix_bits, g_hamming)` where `g` is Hamming distance between the decided hash prefix and the query hash prefix (id bits do not contribute to Hamming).
- Priority: minimize `f = g + h`. Heuristic `h = 0` is admissible (Dijkstra over Hamming); optional `h` = remaining bits that cannot match if known — keep `h=0` for simplicity and correctness first.
- At each depth `< ceil(H/NN)`, expand all `2^NN` labels (or only those Prefix BF accepts — still may false-positive). Prefer checking Prefix BF before enqueue to cut work.
- Stop expanding hash levels when enough terminal candidates collected or a beam/budget (`max_expansions`) is hit (configurable safety valve).
- After hash bits fixed, enumerate id suffixes that pass Prefix + Terminal BF into an `ArrayBuffer` / `Vec<Id>`.
- **Rationale:** user-specified A* + Hamming on vector bits only.

### 6. Reranking
- Map candidate ids back to original float vectors via an id→vector sidecar (`HashMap` or parallel `Vec` built at index time).
- Sort by cosine distance (default); trait `Distance` allows swap-in metrics.
- Return top-K `(id, distance)` ascending distance (nearest first).

### 7. Bloom sizing
- Construct with expected `n` inserts and target false-positive rate `p` (defaults documented, e.g. `p=1e-4`).
- Prefix BF sees more inserts (`~ceil(H/NN)` prefixes per item plus id-path prefixes) — size accordingly.
- **Alternative:** counting Bloom for deletes — out of scope.

### 8. Serialization
- Binary format (versioned magic): header (`version`, `H`, `NN`, `id_width`, `D`, `seed`, Bloom params) + plane coefficients + prefix BF bitset + terminal BF bitset + optional id→vector payload (or separate vectors file).
- Load reconstructs projection + Bloom filters ready for search.
- **Alternative:** only serialize Blooms and require planes regenerated from seed — still need `D` and seed; storing planes avoids float/RNG drift across platforms. Prefer store planes explicitly; seed kept for provenance.

### 9. Testing strategy
- Determinism: fixed seed → identical codes across runs.
- Bloom: insert known keys; positive membership; negative for absent keys (allow rare FP in probabilistic asserts with sized filters).
- Search: tiny handcrafted set where nearest neighbor is known; assert id in top-K.
- Serialize round-trip equality of search results.
- Hamming A*: query identical to an indexed vector yields that id with Hamming 0 path preferred.

## Risks / Trade-offs

- [Bloom false positives] → Spurious trie branches and phantom ids; mitigate with low `p`, Terminal BF confirmation, and cosine rerank on real vectors only (phantoms without sidecar vectors are dropped).
- [Bloom false negatives] → Should not occur for inserted keys if correctly built; mitigate with tests and no post-build mutation corruption.
- [Explosion of `2^NN` branching] → `NN=6` is 64 probes/level; mitigate with Prefix BF gating and expansion budget.
- [Id-bit enumeration cost] → After hash match, many ids may share a hash; mitigate by Prefix BF on `hash||id` prefixes and Terminal BF; document that high collision hash spaces need larger `H`.
- [No deletes] → Document rebuild-for-delete; accept for v1.
- [Cosine on unnormalized vectors] → Define cosine with zero-norm guard (distance = +∞ or skip).

## Migration Plan

- N/A (new crate). Version the on-disk format from `1` so future fields can advance with compatibility checks on load.

## Open Questions

- Exact default false-positive rate and whether vectors are embedded in the same serialized blob or a companion file.
- Whether id width is a compile-time feature flag or runtime enum (`IdWidth::U32 | IdWidth::U64`).
- Whether last partial `NN` chunk (when `H % NN != 0`; `132 % 6 == 0` so default is aligned) should zero-pad — specify pad-with-zeros for non-aligned `H`.
- Beam width / `max_expansions` default for A* under large indexes.
