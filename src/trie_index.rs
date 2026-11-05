use std::collections::HashMap;

use crate::bloom::BloomFilter;
use crate::projection::Projector;
use crate::types::{
    chunk_count, padded_prefix, BitCode, IdWidth, IndexParams,
};

/// Dual Bloom-filter virtual trie plus vector sidecar.
#[derive(Debug, Clone)]
pub struct TrieIndex {
    pub params: IndexParams,
    pub projector: Projector,
    pub prefix_bf: BloomFilter,
    pub terminal_bf: BloomFilter,
    pub vectors: HashMap<i64, Vec<f32>>,
}

impl TrieIndex {
    pub fn new(dim: usize, params: IndexParams, expected_items: usize) -> Self {
        let projector = Projector::new(dim, params.h, params.seed);
        // Prefix inserts: one padded prefix per chunk of the (padded-hash || id) key
        let hash_chunks = chunk_count(params.h, params.nn);
        let total_bits = hash_chunks * params.nn + params.id_width.bits();
        let prefixes_per_item = chunk_count(total_bits, params.nn);
        let prefix_n = expected_items.max(1) * prefixes_per_item;
        let terminal_n = expected_items.max(1);
        Self {
            prefix_bf: BloomFilter::with_rate(prefix_n, params.fp_rate),
            terminal_bf: BloomFilter::with_rate(terminal_n, params.fp_rate),
            params,
            projector,
            vectors: HashMap::new(),
        }
    }

    pub fn from_parts(
        params: IndexParams,
        projector: Projector,
        prefix_bf: BloomFilter,
        terminal_bf: BloomFilter,
        vectors: HashMap<i64, Vec<f32>>,
    ) -> Self {
        Self {
            params,
            projector,
            prefix_bf,
            terminal_bf,
            vectors,
        }
    }

    pub fn dim(&self) -> usize {
        self.projector.dim
    }

    /// Build terminal key = zero-padded hash (to `ceil(H/NN)*NN` bits) || id bits (LE).
    /// Padding keeps search path labels aligned when `H % NN != 0`.
    pub fn terminal_key(&self, hash: &BitCode, id: i64) -> BitCode {
        let nn = self.params.nn;
        let hash_chunks = chunk_count(self.params.h, nn);
        let mut key = padded_prefix(hash, hash_chunks, nn);
        let width = self.params.id_width.bits();
        let raw = match self.params.id_width {
            IdWidth::U32 => (id as u32) as u64,
            IdWidth::U64 => id as u64,
        };
        key.append_bits(raw, width);
        key
    }

    pub fn insert(&mut self, id: i64, vector: Vec<f32>) {
        assert_eq!(vector.len(), self.dim());
        let hash = self.projector.project(&vector);
        let full = self.terminal_key(&hash, id);

        let nn = self.params.nn;
        let hash_chunks = chunk_count(self.params.h, nn);
        let total_chunks = chunk_count(full.bit_len(), nn);

        // Hash-only prefixes (same bytes as full-key prefixes for k <= hash_chunks)
        for k in 1..=hash_chunks {
            let pref = padded_prefix(&hash, k, nn);
            self.prefix_bf.insert_code(&pref);
        }
        // Prefixes spanning into id bits
        for k in (hash_chunks + 1)..=total_chunks {
            let pref = padded_prefix(&full, k, nn);
            self.prefix_bf.insert_code(&pref);
        }

        self.terminal_bf.insert_code(&full);
        self.vectors.insert(id, vector);
    }

    pub fn contains_prefix(&self, prefix: &BitCode) -> bool {
        self.prefix_bf.contains_code(prefix)
    }

    pub fn contains_terminal(&self, key: &BitCode) -> bool {
        self.terminal_bf.contains_code(key)
    }

    pub fn hash_of(&self, vector: &[f32]) -> BitCode {
        self.projector.project(vector)
    }
}

/// Build an index from a dataset of (id, vector) pairs.
pub fn build_from_dataset(
    dataset: &[(i64, Vec<f32>)],
    params: IndexParams,
) -> TrieIndex {
    assert!(!dataset.is_empty());
    let dim = dataset[0].1.len();
    let mut index = TrieIndex::new(dim, params, dataset.len());
    for (id, vec) in dataset {
        index.insert(*id, vec.clone());
    }
    index
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::chunk_count;

    #[test]
    fn insert_positive_probes() {
        let params = IndexParams {
            h: 12,
            nn: 6,
            fp_rate: 1e-6,
            ..IndexParams::default()
        };
        let mut idx = TrieIndex::new(4, params, 8);
        let v = vec![1.0, 0.0, 0.0, 0.0];
        idx.insert(42, v.clone());
        let hash = idx.hash_of(&v);
        let full = idx.terminal_key(&hash, 42);
        assert!(idx.contains_terminal(&full));
        let pref = padded_prefix(&hash, 1, 6);
        assert!(idx.contains_prefix(&pref));
    }

    #[test]
    fn absent_key_negative() {
        let params = IndexParams {
            h: 12,
            nn: 6,
            fp_rate: 1e-6,
            ..IndexParams::default()
        };
        let mut idx = TrieIndex::new(2, params, 4);
        idx.insert(1, vec![1.0, 0.0]);
        let fake = BitCode::zeros(12 + 32);
        assert!(!idx.contains_terminal(&fake));
    }

    #[test]
    fn terminal_key_lengths() {
        let p32 = IndexParams {
            h: 256,
            id_width: IdWidth::U32,
            ..IndexParams::default()
        };
        let idx32 = TrieIndex::new(3, p32, 1);
        let h = BitCode::zeros(256);
        // padded hash = 43*6 = 258, + 32 id bits = 290
        assert_eq!(idx32.terminal_key(&h, 7).bit_len(), 290);

        let p64 = IndexParams {
            h: 256,
            id_width: IdWidth::U64,
            ..IndexParams::default()
        };
        let idx64 = TrieIndex::new(3, p64, 1);
        // padded hash 258 + 64 id bits = 322
        assert_eq!(idx64.terminal_key(&h, 7).bit_len(), 322);
    }

    #[test]
    fn default_nn_h_chunk_count() {
        assert_eq!(chunk_count(256, 6), 43);
        assert_eq!(256 % 6, 4); // last chunk zero-padded
    }
}
