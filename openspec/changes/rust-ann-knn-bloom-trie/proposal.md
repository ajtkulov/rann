## Why

Approximate nearest-neighbor search over high-dimensional vectors needs a compact in-memory index that supports fast candidate retrieval without storing a full dense trie. This change builds a Rust ANN/KNN library that hashes vectors into fixed binary codes via random hyperplanes, indexes them in dual Bloom filters that act as a virtual trie, and retrieves top-K neighbors with A* Hamming search plus cosine reranking.

## What Changes

- Add a Rust crate that projects multi-dimensional float vectors into an `H`-bit binary code using a fixed set of random hyperplanes (seed `1729`, default `H=256`).
- Store each hashed vector plus item id (`int32` or `int64`) as a binary key in an in-memory dual-Bloom-filter trie: a prefix Bloom filter for path existence and a terminal Bloom filter for full `vector+id` membership.
- Split binary keys into `NN`-bit chunks (default `NN=6`) for trie edge labels.
- Implement ANN/KNN search: convert the query to a binary code, A*-traverse the virtual trie guided by prefix Bloom checks while minimizing Hamming distance on the vector bits, collect valid ids via terminal Bloom checks into an array buffer, then rerank and return top-K by cosine distance (configurable distance function).
- Serialize and deserialize Bloom filters for a given input dataset so indexes can be persisted and reloaded.
- Add unit and integration tests covering hashing determinism, Bloom insert/lookup, A* search correctness, serialization round-trips, and top-K ranking.

## Capabilities

### New Capabilities
- `lsh-projection`: Deterministic random-hyperplane LSH that maps float vectors to fixed-length binary codes (`H`, seed).
- `bloom-trie-index`: Dual Bloom-filter virtual trie storing `NN`-bit prefixes and terminal `hash+id` keys.
- `ann-knn-search`: A* Hamming-guided ANN traversal with top-K cosine (or pluggable) distance reranking.
- `index-persistence`: Serialize/deserialize the dual Bloom filters and projection parameters for a dataset.

### Modified Capabilities

- (none — greenfield project)

## Impact

- New Rust library/crate at the repo root (Cargo workspace or single crate); no existing application code.
- Public API surface for build-index, insert, search(top-K), and load/save.
- Dependencies likely include a Bloom filter crate (or custom bitset), RNG with seeded reproducibility, and test fixtures with small synthetic vector sets.
- Memory-bound in-process index; false-positive rates of Bloom filters affect recall and must be documented as tuning parameters.
