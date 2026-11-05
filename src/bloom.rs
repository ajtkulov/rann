use crate::types::BitCode;

/// Classic Bloom filter with double hashing.
#[derive(Debug, Clone)]
pub struct BloomFilter {
    bits: Vec<u64>,
    bit_count: usize,
    hash_count: usize,
    insert_count: usize,
}

impl BloomFilter {
    /// Size for `n` expected inserts and false-positive rate `p`.
    pub fn with_rate(n: usize, p: f64) -> Self {
        assert!(p > 0.0 && p < 1.0);
        let n = n.max(1) as f64;
        let m = (-n * p.ln() / (2.0f64.ln().powi(2))).ceil() as usize;
        let m = m.max(64);
        let k = ((m as f64 / n) * 2.0f64.ln()).round() as usize;
        let k = k.clamp(1, 32);
        Self::with_size(m, k)
    }

    pub fn with_size(bit_count: usize, hash_count: usize) -> Self {
        let words = bit_count.div_ceil(64);
        Self {
            bits: vec![0; words],
            bit_count,
            hash_count,
            insert_count: 0,
        }
    }

    pub fn bit_count(&self) -> usize {
        self.bit_count
    }

    pub fn hash_count(&self) -> usize {
        self.hash_count
    }

    pub fn insert_count(&self) -> usize {
        self.insert_count
    }

    pub fn insert_bytes(&mut self, key: &[u8]) {
        let (h1, h2) = hash_pair(key);
        for i in 0..self.hash_count {
            let idx = self.nth_index(h1, h2, i);
            self.set(idx);
        }
        self.insert_count += 1;
    }

    pub fn insert_code(&mut self, code: &BitCode) {
        self.insert_bytes(&code.to_bytes());
    }

    pub fn contains_bytes(&self, key: &[u8]) -> bool {
        let (h1, h2) = hash_pair(key);
        for i in 0..self.hash_count {
            let idx = self.nth_index(h1, h2, i);
            if !self.get(idx) {
                return false;
            }
        }
        true
    }

    pub fn contains_code(&self, code: &BitCode) -> bool {
        self.contains_bytes(&code.to_bytes())
    }

    fn nth_index(&self, h1: u64, h2: u64, i: usize) -> usize {
        let mix = h1.wrapping_add((i as u64).wrapping_mul(h2));
        (mix as usize) % self.bit_count
    }

    fn set(&mut self, idx: usize) {
        self.bits[idx / 64] |= 1u64 << (idx % 64);
    }

    fn get(&self, idx: usize) -> bool {
        (self.bits[idx / 64] >> (idx % 64)) & 1 == 1
    }

    pub fn words(&self) -> &[u64] {
        &self.bits
    }

    pub fn from_parts(
        bits: Vec<u64>,
        bit_count: usize,
        hash_count: usize,
        insert_count: usize,
    ) -> Self {
        Self {
            bits,
            bit_count,
            hash_count,
            insert_count,
        }
    }
}

fn hash_pair(key: &[u8]) -> (u64, u64) {
    // FNV-1a 64 + a second mix
    let mut h1: u64 = 0xcbf29ce484222325;
    for &b in key {
        h1 ^= b as u64;
        h1 = h1.wrapping_mul(0x100000001b3);
    }
    let mut h2: u64 = 0x84222325cbf29ce4;
    for &b in key.iter().rev() {
        h2 ^= b as u64;
        h2 = h2.wrapping_mul(0x100000001b3);
    }
    if h2 == 0 {
        h2 = 0x9e3779b97f4a7c15;
    }
    (h1, h2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_contains() {
        let mut bf = BloomFilter::with_rate(100, 1e-4);
        let key = b"hello";
        bf.insert_bytes(key);
        assert!(bf.contains_bytes(key));
        assert!(!bf.contains_bytes(b"world-absent-key-xyz"));
    }
}
