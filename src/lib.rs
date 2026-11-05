//! `rann` — approximate nearest-neighbor search using random-hyperplane LSH
//! and a dual Bloom-filter virtual trie with A* Hamming search + cosine rerank.

pub mod benchmark;
pub mod bloom;
pub mod distance;
pub mod projection;
pub mod search;
pub mod serde_index;
pub mod trie_index;
pub mod types;

pub use distance::{CosineDistance, Distance};
pub use projection::Projector;
pub use serde_index::SerdeError;
pub use trie_index::{build_from_dataset, TrieIndex};
pub use types::{BitCode, IdWidth, IndexParams};

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn end_to_end_build_search_save_load() {
        let params = IndexParams {
            h: 36,
            nn: 6,
            seed: 1729,
            fp_rate: 1e-6,
            max_expansions: 100_000,
            id_width: IdWidth::U32,
        };
        let data: Vec<(i64, Vec<f32>)> = (0..30)
            .map(|i| {
                let angle = (i as f32) * 0.2;
                (i, vec![angle.cos(), angle.sin(), 0.1 * i as f32])
            })
            .collect();

        let index = build_from_dataset(&data, params);
        let query = data[5].1.clone();
        let hits = index.search(&query, 5);
        assert!(!hits.is_empty());
        assert_eq!(hits[0].0, 5);

        let mut blob = Vec::new();
        index.save(&mut blob).unwrap();
        let loaded = TrieIndex::load(&mut blob.as_slice()).unwrap();
        let hits2 = loaded.search(&query, 5);
        assert_eq!(hits, hits2);
    }
}
