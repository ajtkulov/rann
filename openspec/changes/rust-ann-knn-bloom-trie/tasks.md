## 1. Crate scaffolding

- [x] 1.1 Initialize Rust library crate `rann` with module stubs: `projection`, `bloom`, `trie_index`, `search`, `distance`, `serde_index`
- [x] 1.2 Add dependencies (seeded RNG, bitvec/bitset helpers, serde or custom binary IO as chosen) and wire `lib.rs` exports
- [x] 1.3 Define shared types: `IdWidth` (U32/U64), `IndexParams { H, NN, seed, fp_rate }`, packed `BitCode` buffer

## 2. LSH projection

- [x] 2.1 Implement seeded hyperplane generator for dimension `D` and count `H` (default seed `1729`, `H=132`)
- [x] 2.2 Implement `project(vector) -> BitCode` using sign of dot products
- [x] 2.3 Expose plane coefficients for persistence and reconstruct projector from stored planes
- [x] 2.4 Tests: determinism for same seed; bit polarity for synthetic half-spaces; default `H=132`

## 3. Bloom filters and virtual trie

- [x] 3.1 Implement Bloom filter (or wrap crate) with sizing from expected inserts + false-positive rate
- [x] 3.2 Implement `NN`-bit chunking with zero-padding on the final partial chunk
- [x] 3.3 Implement insert: write all hash prefixes and `hash||id` prefixes into prefix BF; write full `hash||id` into terminal BF; store id→vector sidecar
- [x] 3.4 Implement prefix and terminal membership probes
- [x] 3.5 Tests: insert then positive probes; absent keys negative at low FP rate; key lengths for i32/i64; default `NN=6` and `H=132` alignment

## 4. Distance and candidate ranking

- [x] 4.1 Implement `Distance` trait with cosine distance (zero-norm guard)
- [x] 4.2 Implement top-K selection over `(id, vector)` candidates vs query
- [x] 4.3 Tests: cosine ordering on known vectors; `K` larger than candidate set; drop ids missing from sidecar

## 5. A* ANN search

- [x] 5.1 Implement A* state over trie depth with Hamming `g` on hash bits only
- [x] 5.2 Expand `2^NN` edge labels gated by prefix Bloom; prefer lower Hamming
- [x] 5.3 After `H` hash bits, enumerate id suffixes via prefix + terminal Bloom into an id array buffer
- [x] 5.4 Enforce configurable `max_expansions` budget; rerank collected candidates with cosine (default)
- [x] 5.5 Public `search(query, K) -> Vec<(Id, distance)>` API
- [x] 5.6 Tests: identical query finds its own id; Hamming preference on tiny fixture; budget stop still returns partial results; top-K cosine order

## 6. Serialization

- [x] 6.1 Define versioned binary format (magic, version, params, planes, both Bloom bitsets, vector sidecar)
- [x] 6.2 Implement `save` / `load` round-trip
- [x] 6.3 Reject unsupported versions with error
- [x] 6.4 Tests: serialize→deserialize preserves search results on a fixed dataset; bad version errors

## 7. Dataset build workflow and integration tests

- [x] 7.1 Implement build-from-dataset `(id, vector)[]` → index → optional serialize
- [x] 7.2 End-to-end integration test: build, search top-K, save, load, search again with matching results
- [x] 7.3 README sketch: hyperparameters (`H`, `NN`, seed, distance) and how to run tests
