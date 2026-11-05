//! Hamming-guided Bloom-trie candidate discovery + cosine rerank.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet, VecDeque};

use crate::distance::{top_k_by_distance, CosineDistance, Distance};
use crate::trie_index::TrieIndex;
use crate::types::{chunk_count, extend_prefix, BitCode};

/// Search result: item id and distance to the query.
pub type Neighbor = (i64, f32);

/// How to discover low-Hamming hash paths on the Bloom trie.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HammingStrategy {
    /// Dijkstra / A* with h=0 (exact order among explored paths).
    AStar,
    /// Increasing Hamming radius multiprobe.
    MultiProbe,
    /// Beam search (bounded fringe; not exact).
    Beam,
}

impl HammingStrategy {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "astar" | "a*" | "dijkstra" => Some(Self::AStar),
            "multiprobe" | "multi-probe" | "radius" => Some(Self::MultiProbe),
            "beam" => Some(Self::Beam),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::AStar => "astar",
            Self::MultiProbe => "multiprobe",
            Self::Beam => "beam",
        }
    }
}

/// Parameters for Hamming trie search.
#[derive(Debug, Clone)]
pub struct SearchParams {
    pub strategy: HammingStrategy,
    /// Override index `max_expansions` when set.
    pub max_expansions: Option<usize>,
    /// Multi-probe: maximum Hamming radius.
    pub r_max: u32,
    /// Multi-probe: stop after this many terminal candidates.
    pub min_candidates: usize,
    /// Beam width.
    pub beam_width: usize,
}

impl Default for SearchParams {
    fn default() -> Self {
        Self {
            strategy: HammingStrategy::AStar,
            max_expansions: None,
            r_max: 8,
            min_candidates: 50,
            beam_width: 64,
        }
    }
}

/// Result of candidate discovery (before distance rerank).
#[derive(Debug, Clone)]
pub struct CandidateSet {
    pub ids: Vec<i64>,
    pub expansions: usize,
    pub bloom_probes: usize,
}

#[derive(Debug, Clone)]
struct PathNode {
    g: u32,
    depth: usize,
    prefix: BitCode,
}

impl PartialEq for PathNode {
    fn eq(&self, other: &Self) -> bool {
        self.g == other.g && self.depth == other.depth
    }
}
impl Eq for PathNode {}
impl PartialOrd for PathNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for PathNode {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap by g via reverse ordering
        other
            .g
            .cmp(&self.g)
            .then_with(|| self.depth.cmp(&other.depth))
    }
}

fn chunk_hamming(query_hash: &BitCode, depth: usize, label: u32, nn: usize, h: usize) -> u32 {
    let start = depth * nn;
    let mut add = 0u32;
    for i in 0..nn {
        let bit_i = start + i;
        if bit_i >= h {
            break;
        }
        let child_bit = ((label >> i) & 1) == 1;
        let q_bit = query_hash.get_bit(bit_i);
        if child_bit != q_bit {
            add += 1;
        }
    }
    add
}

fn label_ok_pad(depth: usize, label: u32, nn: usize, bit_limit: usize) -> bool {
    let start = depth * nn;
    for i in 0..nn {
        let bit_i = start + i;
        if bit_i >= bit_limit && ((label >> i) & 1) == 1 {
            return false;
        }
    }
    true
}

impl TrieIndex {
    /// ANN/KNN search with default A* Hamming strategy + cosine top-K.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<Neighbor> {
        self.search_with_params(query, k, &SearchParams::default(), &CosineDistance)
    }

    pub fn search_with_distance<D: Distance>(
        &self,
        query: &[f32],
        k: usize,
        distance: &D,
    ) -> Vec<Neighbor> {
        self.search_with_params(query, k, &SearchParams::default(), distance)
    }

    pub fn search_with_strategy(
        &self,
        query: &[f32],
        k: usize,
        strategy: HammingStrategy,
    ) -> Vec<Neighbor> {
        let mut params = SearchParams::default();
        params.strategy = strategy;
        self.search_with_params(query, k, &params, &CosineDistance)
    }

    pub fn search_with_params<D: Distance>(
        &self,
        query: &[f32],
        k: usize,
        params: &SearchParams,
        distance: &D,
    ) -> Vec<Neighbor> {
        assert_eq!(query.len(), self.dim());
        let qhash = self.projector.project(query);
        let cands = self.discover_candidates(&qhash, params);
        top_k_by_distance(query, &cands.ids, &self.vectors, k, distance)
    }

    /// Discover candidate ids; also returns work counters for benchmarking.
    pub fn discover_candidates(&self, query_hash: &BitCode, params: &SearchParams) -> CandidateSet {
        match params.strategy {
            HammingStrategy::AStar => self.discover_astar(query_hash, params),
            HammingStrategy::MultiProbe => self.discover_multiprobe(query_hash, params),
            HammingStrategy::Beam => self.discover_beam(query_hash, params),
        }
    }

    /// Backward-compatible alias.
    pub fn collect_candidates(&self, query_hash: &BitCode) -> Vec<i64> {
        self.discover_candidates(query_hash, &SearchParams::default())
            .ids
    }

    fn expansion_limit(&self, params: &SearchParams) -> usize {
        params
            .max_expansions
            .unwrap_or(self.params.max_expansions)
    }

    fn hash_layout(&self) -> (usize, usize, usize, u32, usize) {
        let nn = self.params.nn;
        let h = self.params.h;
        let hash_chunks = chunk_count(h, nn);
        // Terminal keys zero-pad the hash to hash_chunks*nn before appending id bits.
        let total_bits = hash_chunks * nn + self.params.id_width.bits();
        let total_chunks = chunk_count(total_bits, nn);
        let max_label = 1u32 << nn;
        (nn, h, hash_chunks, max_label, total_chunks)
    }

    fn discover_astar(&self, query_hash: &BitCode, params: &SearchParams) -> CandidateSet {
        let (nn, h, hash_chunks, max_label, total_chunks) = self.hash_layout();
        let limit = self.expansion_limit(params);

        let mut heap = BinaryHeap::new();
        heap.push(PathNode {
            g: 0,
            depth: 0,
            prefix: BitCode::zeros(0),
        });

        let mut expansions = 0usize;
        let mut bloom_probes = 0usize;
        let mut candidates: Vec<i64> = Vec::new();
        let mut seen_ids: HashSet<i64> = HashSet::new();
        let mut visited: HashSet<(usize, Vec<u8>)> = HashSet::new();

        while let Some(node) = heap.pop() {
            if expansions >= limit {
                break;
            }
            expansions += 1;

            let key = (node.depth, node.prefix.to_bytes());
            if !visited.insert(key) {
                continue;
            }

            if node.depth == hash_chunks {
                self.enumerate_ids(
                    &node.prefix,
                    hash_chunks,
                    total_chunks,
                    nn,
                    max_label,
                    &mut candidates,
                    &mut seen_ids,
                    &mut expansions,
                    &mut bloom_probes,
                    limit,
                );
                continue;
            }
            if node.depth > hash_chunks {
                continue;
            }

            for label in 0..max_label {
                if !label_ok_pad(node.depth, label, nn, h) {
                    continue;
                }
                let child = extend_prefix(&node.prefix, label, nn);
                bloom_probes += 1;
                if !self.contains_prefix(&child) {
                    continue;
                }
                let add = chunk_hamming(query_hash, node.depth, label, nn, h);
                heap.push(PathNode {
                    g: node.g + add,
                    depth: node.depth + 1,
                    prefix: child,
                });
            }
        }

        CandidateSet {
            ids: candidates,
            expansions,
            bloom_probes,
        }
    }

    fn discover_multiprobe(&self, query_hash: &BitCode, params: &SearchParams) -> CandidateSet {
        let (nn, h, hash_chunks, max_label, total_chunks) = self.hash_layout();
        let limit = self.expansion_limit(params);

        let mut expansions = 0usize;
        let mut bloom_probes = 0usize;
        let mut candidates: Vec<i64> = Vec::new();
        let mut seen_ids: HashSet<i64> = HashSet::new();
        // Avoid redoing the same hash path across radii
        let mut finished_hashes: HashSet<Vec<u8>> = HashSet::new();

        for radius in 0..=params.r_max {
            if candidates.len() >= params.min_candidates || expansions >= limit {
                break;
            }

            let mut queue: VecDeque<PathNode> = VecDeque::new();
            queue.push_back(PathNode {
                g: 0,
                depth: 0,
                prefix: BitCode::zeros(0),
            });
            let mut visited: HashSet<(usize, Vec<u8>)> = HashSet::new();

            while let Some(node) = queue.pop_front() {
                if expansions >= limit || candidates.len() >= params.min_candidates {
                    break;
                }
                expansions += 1;

                let key = (node.depth, node.prefix.to_bytes());
                if !visited.insert(key) {
                    continue;
                }

                if node.depth == hash_chunks {
                    if node.g > radius {
                        continue;
                    }
                    let hb = node.prefix.prefix(h).to_bytes();
                    if !finished_hashes.insert(hb) {
                        continue;
                    }
                    self.enumerate_ids(
                        &node.prefix,
                        hash_chunks,
                        total_chunks,
                        nn,
                        max_label,
                        &mut candidates,
                        &mut seen_ids,
                        &mut expansions,
                        &mut bloom_probes,
                        limit,
                    );
                    continue;
                }

                let remaining = radius.saturating_sub(node.g);
                for label in 0..max_label {
                    if !label_ok_pad(node.depth, label, nn, h) {
                        continue;
                    }
                    let add = chunk_hamming(query_hash, node.depth, label, nn, h);
                    if add > remaining {
                        continue;
                    }
                    let child = extend_prefix(&node.prefix, label, nn);
                    bloom_probes += 1;
                    if !self.contains_prefix(&child) {
                        continue;
                    }
                    queue.push_back(PathNode {
                        g: node.g + add,
                        depth: node.depth + 1,
                        prefix: child,
                    });
                }
            }
        }

        CandidateSet {
            ids: candidates,
            expansions,
            bloom_probes,
        }
    }

    fn discover_beam(&self, query_hash: &BitCode, params: &SearchParams) -> CandidateSet {
        let (nn, h, hash_chunks, max_label, total_chunks) = self.hash_layout();
        let limit = self.expansion_limit(params);
        let w = params.beam_width.max(1);

        let mut expansions = 0usize;
        let mut bloom_probes = 0usize;
        let mut candidates: Vec<i64> = Vec::new();
        let mut seen_ids: HashSet<i64> = HashSet::new();

        let mut beam = vec![PathNode {
            g: 0,
            depth: 0,
            prefix: BitCode::zeros(0),
        }];

        for _depth in 0..hash_chunks {
            if expansions >= limit {
                break;
            }
            let mut next: Vec<PathNode> = Vec::new();
            for node in &beam {
                if expansions >= limit {
                    break;
                }
                expansions += 1;
                for label in 0..max_label {
                    if !label_ok_pad(node.depth, label, nn, h) {
                        continue;
                    }
                    let child = extend_prefix(&node.prefix, label, nn);
                    bloom_probes += 1;
                    if !self.contains_prefix(&child) {
                        continue;
                    }
                    let add = chunk_hamming(query_hash, node.depth, label, nn, h);
                    next.push(PathNode {
                        g: node.g + add,
                        depth: node.depth + 1,
                        prefix: child,
                    });
                }
            }
            next.sort_by(|a, b| a.g.cmp(&b.g).then_with(|| a.depth.cmp(&b.depth)));
            // Dedup by prefix bytes keeping best g
            let mut seen_pref = HashSet::new();
            let mut dedup = Vec::new();
            for n in next {
                let key = n.prefix.to_bytes();
                if seen_pref.insert(key) {
                    dedup.push(n);
                }
            }
            dedup.truncate(w);
            beam = dedup;
            if beam.is_empty() {
                break;
            }
        }

        for node in &beam {
            if expansions >= limit {
                break;
            }
            if node.depth == hash_chunks {
                self.enumerate_ids(
                    &node.prefix,
                    hash_chunks,
                    total_chunks,
                    nn,
                    max_label,
                    &mut candidates,
                    &mut seen_ids,
                    &mut expansions,
                    &mut bloom_probes,
                    limit,
                );
            }
        }

        CandidateSet {
            ids: candidates,
            expansions,
            bloom_probes,
        }
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
        bloom_probes: &mut usize,
        limit: usize,
    ) {
        let mut stack = vec![(hash_chunks, hash_prefix.clone())];
        while let Some((depth, prefix)) = stack.pop() {
            if *expansions >= limit {
                return;
            }
            *expansions += 1;

            if depth == total_chunks {
                if let Some(id) = self.id_from_full_prefix(&prefix) {
                    let hash = prefix.prefix(self.params.h);
                    let term = self.terminal_key(&hash, id);
                    *bloom_probes += 1;
                    if self.contains_terminal(&term) && seen.insert(id) {
                        out.push(id);
                    }
                }
                continue;
            }

            let padded_h = hash_chunks * nn;
            let total_bits = padded_h + self.params.id_width.bits();
            for label in 0..max_label {
                if !label_ok_pad(depth, label, nn, total_bits) {
                    continue;
                }
                let child = extend_prefix(&prefix, label, nn);
                *bloom_probes += 1;
                if self.contains_prefix(&child) {
                    stack.push((depth + 1, child));
                }
            }
        }
    }

    fn id_from_full_prefix(&self, prefix: &BitCode) -> Option<i64> {
        let nn = self.params.nn;
        let id_bits = self.params.id_width.bits();
        let id_start = chunk_count(self.params.h, nn) * nn;
        if prefix.bit_len() < id_start {
            return None;
        }
        let mut raw = 0u64;
        for i in 0..id_bits {
            let idx = id_start + i;
            if idx < prefix.bit_len() && prefix.get_bit(idx) {
                raw |= 1u64 << i;
            }
        }
        Some(raw as i64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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

    fn tiny_index() -> TrieIndex {
        let data = vec![
            (1, vec![1.0, 0.0, 0.0]),
            (2, vec![0.0, 1.0, 0.0]),
            (3, vec![0.0, 0.0, 1.0]),
            (4, vec![0.9, 0.1, 0.0]),
            (5, vec![0.1, 0.9, 0.0]),
        ];
        build_from_dataset(&data, tiny_params())
    }

    #[test]
    fn identical_query_finds_self() {
        let idx = tiny_index();
        let hits = idx.search(&[1.0, 0.0, 0.0], 1);
        assert_eq!(hits[0].0, 1);
    }

    #[test]
    fn unaligned_h_finds_self_all_strategies() {
        // H=256, NN=6 → H % NN != 0 (the GloVe default); must not over-count id chunks.
        let params = IndexParams {
            h: 256,
            nn: 6,
            seed: 1729,
            fp_rate: 1e-6,
            max_expansions: 50_000,
            ..IndexParams::default()
        };
        let data = vec![
            (1, vec![1.0, 0.0, 0.0, 0.0]),
            (2, vec![0.0, 1.0, 0.0, 0.0]),
            (3, vec![0.0, 0.0, 1.0, 0.0]),
        ];
        let idx = build_from_dataset(&data, params);
        let q = vec![1.0, 0.0, 0.0, 0.0];
        for strategy in [
            HammingStrategy::AStar,
            HammingStrategy::MultiProbe,
            HammingStrategy::Beam,
        ] {
            let mut sp = SearchParams::default();
            sp.strategy = strategy;
            sp.min_candidates = 1;
            sp.r_max = 4;
            sp.beam_width = 32;
            let hits = idx.search_with_params(&q, 1, &sp, &CosineDistance);
            assert_eq!(
                hits[0].0, 1,
                "strategy {} missed self under unaligned H",
                strategy.as_str()
            );
        }
    }

    #[test]
    fn all_strategies_find_self() {
        let idx = tiny_index();
        let q = vec![1.0, 0.0, 0.0];
        for strategy in [
            HammingStrategy::AStar,
            HammingStrategy::MultiProbe,
            HammingStrategy::Beam,
        ] {
            let mut params = SearchParams::default();
            params.strategy = strategy;
            params.min_candidates = 1;
            params.r_max = 16;
            params.beam_width = 64;
            let hits = idx.search_with_params(&q, 3, &params, &CosineDistance);
            assert!(
                hits.iter().any(|(id, _)| *id == 1),
                "strategy {} missed self: {:?}",
                strategy.as_str(),
                hits
            );
        }
    }

    #[test]
    fn multiprobe_prefers_radius_zero_path() {
        // Smoke: with r_max=0 we still run without panic and expansions > 0
        let idx = tiny_index();
        let qhash = idx.projector.project(&[1.0, 0.0, 0.0]);
        let mut params = SearchParams::default();
        params.strategy = HammingStrategy::MultiProbe;
        params.r_max = 0;
        params.min_candidates = 100;
        let c = idx.discover_candidates(&qhash, &params);
        assert!(c.expansions > 0);
    }

    #[test]
    fn beam_respects_width_bound_on_work() {
        let idx = tiny_index();
        let qhash = idx.projector.project(&[1.0, 0.0, 0.0]);
        let mut narrow = SearchParams::default();
        narrow.strategy = HammingStrategy::Beam;
        narrow.beam_width = 1;
        let mut wide = SearchParams::default();
        wide.strategy = HammingStrategy::Beam;
        wide.beam_width = 64;
        let a = idx.discover_candidates(&qhash, &narrow);
        let b = idx.discover_candidates(&qhash, &wide);
        // Wider beam should not use fewer bloom probes in typical cases
        assert!(b.bloom_probes >= a.bloom_probes || b.ids.len() >= a.ids.len());
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
        let _ = idx.search(&data[0].1, 3);
    }
}
