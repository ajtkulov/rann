use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

use crate::distance::{top_k_by_distance, CosineDistance, Distance};
use crate::trie_index::TrieIndex;
use crate::types::{chunk_count, extend_prefix, BitCode};

#[derive(Debug, Clone)]
struct AstNode {
    /// Hamming distance on hash bits decided so far.
    g: u32,
    /// Number of NN-chunks decided (hash phase then id phase).
    depth: usize,
    /// Bits decided so far (multiple of nn, zero-padded within last chunk).
    prefix: BitCode,
}

impl PartialEq for AstNode {
    fn eq(&self, other: &Self) -> bool {
        self.g == other.g && self.depth == other.depth
    }
}
impl Eq for AstNode {}

impl PartialOrd for AstNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// Min-heap by g, then deeper first as tie-break.
impl Ord for AstNode {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .g
            .cmp(&self.g)
            .then_with(|| self.depth.cmp(&other.depth))
    }
}

/// Search result: item id and distance to the query.
pub type Neighbor = (i64, f32);

impl TrieIndex {
    /// ANN/KNN search: A* over the Bloom trie (Hamming on hash bits), then cosine top-K.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<Neighbor> {
        self.search_with_distance(query, k, &CosineDistance)
    }

    pub fn search_with_distance<D: Distance>(
        &self,
        query: &[f32],
        k: usize,
        distance: &D,
    ) -> Vec<Neighbor> {
        assert_eq!(query.len(), self.dim());
        let qhash = self.projector.project(query);
        let ids = self.collect_candidates(&qhash);
        top_k_by_distance(query, &ids, &self.vectors, k, distance)
    }

    /// Collect candidate ids via A* + id-suffix enumeration into an array buffer.
    pub fn collect_candidates(&self, query_hash: &BitCode) -> Vec<i64> {
        let nn = self.params.nn;
        let h = self.params.h;
        let hash_chunks = chunk_count(h, nn);
        let id_bits = self.params.id_width.bits();
        let id_chunks = chunk_count(id_bits, nn);
        let total_chunks = hash_chunks + id_chunks;
        let max_label = 1u32 << nn;

        let mut heap = BinaryHeap::new();
        heap.push(AstNode {
            g: 0,
            depth: 0,
            prefix: BitCode::zeros(0),
        });

        let mut expansions = 0usize;
        let mut candidates: Vec<i64> = Vec::new();
        let mut seen_ids: HashSet<i64> = HashSet::new();
        // Track visited (depth, prefix bytes) lightly to avoid re-expanding
        let mut visited: HashSet<(usize, Vec<u8>)> = HashSet::new();

        while let Some(node) = heap.pop() {
            if expansions >= self.params.max_expansions {
                break;
            }
            expansions += 1;

            let key = (node.depth, node.prefix.to_bytes());
            if !visited.insert(key) {
                continue;
            }

            if node.depth == hash_chunks {
                // Enumerate id suffixes from this hash path
                self.enumerate_ids(
                    &node.prefix,
                    hash_chunks,
                    total_chunks,
                    nn,
                    max_label,
                    &mut candidates,
                    &mut seen_ids,
                    &mut expansions,
                );
                continue;
            }

            if node.depth > hash_chunks {
                continue;
            }

            // Expand all NN-bit labels gated by prefix Bloom
            let start = node.depth * nn;
            for label in 0..max_label {
                // Force zero-padding past H on the final hash chunk
                let mut ok_pad = true;
                for i in 0..nn {
                    let bit_i = start + i;
                    if bit_i >= h && ((label >> i) & 1) == 1 {
                        ok_pad = false;
                        break;
                    }
                }
                if !ok_pad {
                    continue;
                }

                let child = extend_prefix(&node.prefix, label, nn);
                if !self.contains_prefix(&child) {
                    continue;
                }
                // Hamming increment for this chunk vs query hash (hash bits only)
                let mut add = 0u32;
                for i in 0..nn {
                    let bit_i = start + i;
                    if bit_i >= h {
                        break;
                    }
                    let child_bit = child.get_bit(bit_i);
                    let q_bit = query_hash.get_bit(bit_i);
                    if child_bit != q_bit {
                        add += 1;
                    }
                }
                heap.push(AstNode {
                    g: node.g + add,
                    depth: node.depth + 1,
                    prefix: child,
                });
            }
        }

        candidates
    }

    #[allow(clippy::too_many_arguments)]
    fn enumerate_ids(
        &self,
        hash_prefix: &BitCode,
        hash_chunks: usize,
        total_chunks: usize,
        nn: usize,
        max_label: u32,
        out: &mut Vec<i64>,
        seen: &mut HashSet<i64>,
        expansions: &mut usize,
    ) {
        // BFS over id chunks; Hamming no longer grows
        let mut stack = vec![(hash_chunks, hash_prefix.clone())];
        while let Some((depth, prefix)) = stack.pop() {
            if *expansions >= self.params.max_expansions {
                return;
            }
            *expansions += 1;

            if depth == total_chunks {
                if let Some(id) = self.id_from_full_prefix(&prefix) {
                    let hash = prefix.prefix(self.params.h);
                    let term = self.terminal_key(&hash, id);
                    if self.contains_terminal(&term) && seen.insert(id) {
                        out.push(id);
                    }
                }
                continue;
            }

            let start = depth * nn;
            let total_bits = self.params.h + self.params.id_width.bits();
            for label in 0..max_label {
                let mut ok_pad = true;
                for i in 0..nn {
                    let bit_i = start + i;
                    if bit_i >= total_bits && ((label >> i) & 1) == 1 {
                        ok_pad = false;
                        break;
                    }
                }
                if !ok_pad {
                    continue;
                }
                let child = extend_prefix(&prefix, label, nn);
                if self.contains_prefix(&child) {
                    stack.push((depth + 1, child));
                }
            }
        }
    }

    fn id_from_full_prefix(&self, prefix: &BitCode) -> Option<i64> {
        let h = self.params.h;
        let id_bits = self.params.id_width.bits();
        if prefix.bit_len() < h {
            return None;
        }
        let mut raw = 0u64;
        for i in 0..id_bits {
            let idx = h + i;
            if idx < prefix.bit_len() && prefix.get_bit(idx) {
                raw |= 1u64 << i;
            }
        }
        Some(raw as i64)
    }
}

#[cfg(test)]
mod tests {
    use crate::trie_index::build_from_dataset;
    use crate::types::IndexParams;

    fn tiny_params() -> IndexParams {
        IndexParams {
            h: 24,
            nn: 6,
            seed: 1729,
            fp_rate: 1e-6,
            max_expansions: 50_000,
            ..IndexParams::default()
        }
    }

    #[test]
    fn identical_query_finds_self() {
        let data = vec![
            (1, vec![1.0, 0.0, 0.0]),
            (2, vec![0.0, 1.0, 0.0]),
            (3, vec![0.0, 0.0, 1.0]),
        ];
        let idx = build_from_dataset(&data, tiny_params());
        let hits = idx.search(&data[0].1, 1);
        assert_eq!(hits[0].0, 1);
    }

    #[test]
    fn top_k_cosine_order() {
        let data = vec![
            (1, vec![1.0, 0.0]),
            (2, vec![0.9, 0.1]),
            (3, vec![0.0, 1.0]),
        ];
        let idx = build_from_dataset(&data, tiny_params());
        let hits = idx.search(&[1.0, 0.0], 3);
        assert_eq!(hits[0].0, 1);
        assert!(hits.len() <= 3);
    }

    #[test]
    fn budget_still_returns() {
        let data: Vec<_> = (0..20)
            .map(|i| (i, vec![i as f32, (20 - i) as f32, 0.5]))
            .collect();
        let mut params = tiny_params();
        params.max_expansions = 5;
        let idx = build_from_dataset(&data, params);
        let _ = idx.search(&data[0].1, 3); // must not panic
    }
}
