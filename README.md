# rann

Approximate nearest-neighbor (ANN/KNN) search in Rust using:

1. **Random-hyperplane LSH** — maps float vectors to an `H`-bit code (default `H=132`, seed `1729`)
2. **Dual Bloom-filter virtual trie** — prefix + terminal filters over `NN`-bit edges (default `NN=6`)
3. **A\* search** — expands edges while minimizing Hamming distance on hash bits
4. **Cosine rerank** — returns top-K by cosine distance (pluggable)

## Quick start

```rust
use rann::{build_from_dataset, IndexParams};

let data = vec![
    (1_i64, vec![1.0, 0.0, 0.0]),
    (2, vec![0.0, 1.0, 0.0]),
];
let index = build_from_dataset(&data, IndexParams::default());
let neighbors = index.search(&[1.0, 0.0, 0.0], 1);
```

Persist with `index.save(&mut writer)?` and reload with `TrieIndex::load(&mut reader)?`.

## Hyperparameters (`IndexParams`)

| Field | Default | Meaning |
|-------|---------|---------|
| `h` | `132` | LSH code length (bits) |
| `nn` | `6` | Trie edge width (bits); `2^nn` labels per depth |
| `seed` | `1729` | RNG seed for random hyperplanes |
| `fp_rate` | `1e-4` | Bloom filter target false-positive rate |
| `id_width` | `U32` | Item id bits appended to the terminal key (`U32` or `U64`) |
| `max_expansions` | `100000` | A\* expansion budget |

Distance defaults to cosine (`1 - cos`); pass another `Distance` via `search_with_distance`.

## Build a dataset from GloVe

```bash
# Full GloVe 6B 300d (400k vectors) → data/glove.6B.300d.rann + .vocab
cargo run --release --bin gen_ds -- glove.6B.300d.txt

# Smoke / subset
cargo run --release --bin gen_ds -- glove.6B.300d.txt --limit 10000 -o data/glove10k
```

## Tests

```bash
cargo test
```
