/// Pluggable distance: smaller is nearer.
pub trait Distance: Send + Sync {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32;
}

/// Cosine distance = 1 - cosine similarity. Zero-norm vectors yield +∞.
#[derive(Debug, Clone, Copy, Default)]
pub struct CosineDistance;

impl Distance for CosineDistance {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        let mut dot = 0.0f32;
        let mut na = 0.0f32;
        let mut nb = 0.0f32;
        for (x, y) in a.iter().zip(b.iter()) {
            dot += x * y;
            na += x * x;
            nb += y * y;
        }
        if na == 0.0 || nb == 0.0 {
            return f32::INFINITY;
        }
        let cos = dot / (na.sqrt() * nb.sqrt());
        // Clamp numeric noise into [-1, 1]
        let cos = cos.clamp(-1.0, 1.0);
        1.0 - cos
    }
}

/// Select top-K nearest among candidates that have vectors in `lookup`.
pub fn top_k_by_distance<D: Distance>(
    query: &[f32],
    candidate_ids: &[i64],
    lookup: &std::collections::HashMap<i64, Vec<f32>>,
    k: usize,
    distance: &D,
) -> Vec<(i64, f32)> {
    let mut scored: Vec<(i64, f32)> = candidate_ids
        .iter()
        .filter_map(|&id| {
            let v = lookup.get(&id)?;
            Some((id, distance.distance(query, v)))
        })
        .collect();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    if scored.len() > k {
        scored.truncate(k);
    }
    scored
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn cosine_ordering() {
        let q = vec![1.0, 0.0];
        let mut map = HashMap::new();
        map.insert(1, vec![1.0, 0.0]); // identical
        map.insert(2, vec![0.0, 1.0]); // orthogonal
        map.insert(3, vec![-1.0, 0.0]); // opposite
        let ids = vec![1, 2, 3];
        let top = top_k_by_distance(&q, &ids, &map, 3, &CosineDistance);
        assert_eq!(top[0].0, 1);
        assert_eq!(top[1].0, 2);
        assert_eq!(top[2].0, 3);
    }

    #[test]
    fn k_larger_than_set() {
        let q = vec![1.0];
        let mut map = HashMap::new();
        map.insert(7, vec![1.0]);
        let top = top_k_by_distance(&q, &[7], &map, 10, &CosineDistance);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn drop_missing_sidecar() {
        let q = vec![1.0, 0.0];
        let map = HashMap::new();
        let top = top_k_by_distance(&q, &[99], &map, 5, &CosineDistance);
        assert!(top.is_empty());
    }

    #[test]
    fn zero_norm_infinite() {
        let d = CosineDistance.distance(&[0.0, 0.0], &[1.0, 0.0]);
        assert!(d.is_infinite());
    }
}
