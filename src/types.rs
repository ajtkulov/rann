/// How many bits of the item id are appended to the hash in terminal keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdWidth {
    U32,
    U64,
}

impl IdWidth {
    pub fn bits(self) -> usize {
        match self {
            IdWidth::U32 => 32,
            IdWidth::U64 => 64,
        }
    }

    pub fn as_u8(self) -> u8 {
        match self {
            IdWidth::U32 => 32,
            IdWidth::U64 => 64,
        }
    }

    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            32 => Some(IdWidth::U32),
            64 => Some(IdWidth::U64),
            _ => None,
        }
    }
}

/// Packed bit buffer used for LSH codes and Bloom keys.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BitCode {
    words: Vec<u64>,
    bit_len: usize,
}

impl BitCode {
    pub fn zeros(bit_len: usize) -> Self {
        let n_words = bit_len.div_ceil(64);
        Self {
            words: vec![0; n_words],
            bit_len,
        }
    }

    pub fn bit_len(&self) -> usize {
        self.bit_len
    }

    pub fn get_bit(&self, i: usize) -> bool {
        assert!(i < self.bit_len);
        (self.words[i / 64] >> (i % 64)) & 1 == 1
    }

    pub fn set_bit(&mut self, i: usize, value: bool) {
        assert!(i < self.bit_len);
        let w = i / 64;
        let b = i % 64;
        if value {
            self.words[w] |= 1u64 << b;
        } else {
            self.words[w] &= !(1u64 << b);
        }
    }

    /// Append `count` bits from `value` (low bit first / little-endian bit order).
    pub fn append_bits(&mut self, value: u64, count: usize) {
        assert!(count <= 64);
        let start = self.bit_len;
        self.bit_len += count;
        let need = self.bit_len.div_ceil(64);
        if self.words.len() < need {
            self.words.resize(need, 0);
        }
        for i in 0..count {
            let bit = ((value >> i) & 1) == 1;
            let idx = start + i;
            let w = idx / 64;
            let b = idx % 64;
            if bit {
                self.words[w] |= 1u64 << b;
            } else {
                self.words[w] &= !(1u64 << b);
            }
        }
    }

    pub fn concat(a: &BitCode, b: &BitCode) -> BitCode {
        let mut out = a.clone();
        for i in 0..b.bit_len {
            out.append_bits(if b.get_bit(i) { 1 } else { 0 }, 1);
        }
        out
    }

    pub fn prefix(&self, len: usize) -> BitCode {
        assert!(len <= self.bit_len);
        let mut out = BitCode::zeros(len);
        for i in 0..len {
            out.set_bit(i, self.get_bit(i));
        }
        out
    }

    /// Hamming distance on the overlapping prefix (min length).
    pub fn hamming(&self, other: &BitCode) -> u32 {
        let n = self.bit_len.min(other.bit_len);
        let mut d = 0u32;
        for i in 0..n {
            if self.get_bit(i) != other.get_bit(i) {
                d += 1;
            }
        }
        d
    }

    /// Bytes covering all bits (little-endian words), for hashing / serialization of keys.
    pub fn to_bytes(&self) -> Vec<u8> {
        let nbytes = self.bit_len.div_ceil(8);
        let mut bytes = vec![0u8; nbytes];
        for i in 0..self.bit_len {
            if self.get_bit(i) {
                bytes[i / 8] |= 1u8 << (i % 8);
            }
        }
        bytes
    }

    pub fn from_bits(bits: &[bool]) -> Self {
        let mut c = BitCode::zeros(bits.len());
        for (i, &b) in bits.iter().enumerate() {
            c.set_bit(i, b);
        }
        c
    }

    pub fn words(&self) -> &[u64] {
        &self.words
    }

    pub fn from_words(words: Vec<u64>, bit_len: usize) -> Self {
        Self { words, bit_len }
    }
}

/// Index construction / search hyperparameters.
#[derive(Debug, Clone)]
pub struct IndexParams {
    pub h: usize,
    pub nn: usize,
    pub seed: u64,
    pub fp_rate: f64,
    pub id_width: IdWidth,
    pub max_expansions: usize,
}

impl Default for IndexParams {
    fn default() -> Self {
        Self {
            h: 256,
            nn: 6,
            seed: 1729,
            fp_rate: 1e-4,
            id_width: IdWidth::U32,
            max_expansions: 100_000,
        }
    }
}

/// Number of NN-sized chunks needed to cover `bit_len` bits (ceil).
pub fn chunk_count(bit_len: usize, nn: usize) -> usize {
    assert!(nn > 0);
    bit_len.div_ceil(nn)
}

/// Extract chunk `chunk_idx` of width `nn` from `code`, zero-padding past the end.
pub fn chunk_at(code: &BitCode, chunk_idx: usize, nn: usize) -> u32 {
    let start = chunk_idx * nn;
    let mut label = 0u32;
    for i in 0..nn {
        let bit_i = start + i;
        let bit = if bit_i < code.bit_len() {
            code.get_bit(bit_i)
        } else {
            false
        };
        if bit {
            label |= 1u32 << i;
        }
    }
    label
}

/// Build a BitCode prefix of length `(chunk_idx + 1) * nn` by taking bits from `code`
/// and zero-padding as needed (used for Bloom prefix keys).
pub fn padded_prefix(code: &BitCode, num_chunks: usize, nn: usize) -> BitCode {
    let len = num_chunks * nn;
    let mut out = BitCode::zeros(len);
    for i in 0..len {
        let bit = if i < code.bit_len() {
            code.get_bit(i)
        } else {
            false
        };
        out.set_bit(i, bit);
    }
    out
}

/// Append an NN-bit label onto a prefix, returning a new code of length prefix+nn.
pub fn extend_prefix(prefix: &BitCode, label: u32, nn: usize) -> BitCode {
    let mut out = prefix.clone();
    for i in 0..nn {
        out.append_bits(if (label >> i) & 1 == 1 { 1 } else { 0 }, 1);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_count_default_h() {
        // H=256, NN=6 → 43 chunks (last chunk zero-padded; 256 % 6 == 4)
        assert_eq!(256 % 6, 4);
        assert_eq!(chunk_count(256, 6), 43);
    }

    #[test]
    fn append_and_hamming() {
        let mut a = BitCode::zeros(0);
        a.append_bits(0b101, 3);
        let mut b = BitCode::zeros(0);
        b.append_bits(0b100, 3);
        assert_eq!(a.hamming(&b), 1);
    }
}
