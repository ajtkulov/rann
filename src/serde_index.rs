use std::collections::HashMap;
use std::io::{self, Read, Write};

use crate::bloom::BloomFilter;
use crate::projection::Projector;
use crate::trie_index::TrieIndex;
use crate::types::{IdWidth, IndexParams};

const MAGIC: &[u8; 4] = b"RANN";
const VERSION: u32 = 1;

#[derive(Debug)]
pub enum SerdeError {
    Io(io::Error),
    BadMagic,
    UnsupportedVersion(u32),
    Corrupt(&'static str),
}

impl From<io::Error> for SerdeError {
    fn from(e: io::Error) -> Self {
        SerdeError::Io(e)
    }
}

impl std::fmt::Display for SerdeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SerdeError::Io(e) => write!(f, "io error: {e}"),
            SerdeError::BadMagic => write!(f, "bad magic"),
            SerdeError::UnsupportedVersion(v) => write!(f, "unsupported version {v}"),
            SerdeError::Corrupt(m) => write!(f, "corrupt: {m}"),
        }
    }
}

impl std::error::Error for SerdeError {}

fn write_u32(w: &mut impl Write, v: u32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn write_u64(w: &mut impl Write, v: u64) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn write_f64(w: &mut impl Write, v: f64) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn write_f32(w: &mut impl Write, v: f32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn read_u32(r: &mut impl Read) -> io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64(r: &mut impl Read) -> io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn read_f64(r: &mut impl Read) -> io::Result<f64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(f64::from_le_bytes(b))
}

fn read_f32(r: &mut impl Read) -> io::Result<f32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(f32::from_le_bytes(b))
}

fn write_bloom(w: &mut impl Write, bf: &BloomFilter) -> io::Result<()> {
    write_u64(w, bf.bit_count() as u64)?;
    write_u32(w, bf.hash_count() as u32)?;
    write_u64(w, bf.insert_count() as u64)?;
    for word in bf.words() {
        write_u64(w, *word)?;
    }
    Ok(())
}

fn read_bloom(r: &mut impl Read) -> Result<BloomFilter, SerdeError> {
    let bit_count = read_u64(r)? as usize;
    let hash_count = read_u32(r)? as usize;
    let insert_count = read_u64(r)? as usize;
    let n_words = bit_count.div_ceil(64);
    let mut words = Vec::with_capacity(n_words);
    for _ in 0..n_words {
        words.push(read_u64(r)?);
    }
    Ok(BloomFilter::from_parts(
        words,
        bit_count,
        hash_count,
        insert_count,
    ))
}

impl TrieIndex {
    pub fn save(&self, w: &mut impl Write) -> Result<(), SerdeError> {
        w.write_all(MAGIC)?;
        write_u32(w, VERSION)?;
        write_u32(w, self.params.h as u32)?;
        write_u32(w, self.params.nn as u32)?;
        w.write_all(&[self.params.id_width.as_u8()])?;
        write_u64(w, self.params.seed)?;
        write_f64(w, self.params.fp_rate)?;
        write_u64(w, self.params.max_expansions as u64)?;
        write_u32(w, self.projector.dim as u32)?;

        // planes: h * dim f32
        for plane in &self.projector.planes {
            for &c in plane {
                write_f32(w, c)?;
            }
        }

        write_bloom(w, &self.prefix_bf)?;
        write_bloom(w, &self.terminal_bf)?;

        write_u64(w, self.vectors.len() as u64)?;
        for (&id, vec) in &self.vectors {
            write_u64(w, id as u64)?;
            write_u32(w, vec.len() as u32)?;
            for &x in vec {
                write_f32(w, x)?;
            }
        }
        Ok(())
    }

    pub fn load(r: &mut impl Read) -> Result<Self, SerdeError> {
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(SerdeError::BadMagic);
        }
        let version = read_u32(r)?;
        if version != VERSION {
            return Err(SerdeError::UnsupportedVersion(version));
        }

        let h = read_u32(r)? as usize;
        let nn = read_u32(r)? as usize;
        let mut idw = [0u8; 1];
        r.read_exact(&mut idw)?;
        let id_width =
            IdWidth::from_u8(idw[0]).ok_or(SerdeError::Corrupt("id_width"))?;
        let seed = read_u64(r)?;
        let fp_rate = read_f64(r)?;
        let max_expansions = read_u64(r)? as usize;
        let dim = read_u32(r)? as usize;

        let mut planes = Vec::with_capacity(h);
        for _ in 0..h {
            let mut plane = Vec::with_capacity(dim);
            for _ in 0..dim {
                plane.push(read_f32(r)?);
            }
            planes.push(plane);
        }

        let prefix_bf = read_bloom(r)?;
        let terminal_bf = read_bloom(r)?;

        let n_vecs = read_u64(r)? as usize;
        let mut vectors = HashMap::with_capacity(n_vecs);
        for _ in 0..n_vecs {
            let id = read_u64(r)? as i64;
            let len = read_u32(r)? as usize;
            let mut v = Vec::with_capacity(len);
            for _ in 0..len {
                v.push(read_f32(r)?);
            }
            vectors.insert(id, v);
        }

        let params = IndexParams {
            h,
            nn,
            seed,
            fp_rate,
            id_width,
            max_expansions,
        };
        let projector = Projector::from_planes(dim, h, seed, planes);
        Ok(TrieIndex::from_parts(
            params, projector, prefix_bf, terminal_bf, vectors,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trie_index::build_from_dataset;
    use crate::types::IndexParams;

    #[test]
    fn round_trip_preserves_search() {
        let params = IndexParams {
            h: 24,
            nn: 6,
            fp_rate: 1e-6,
            ..IndexParams::default()
        };
        let data = vec![
            (1, vec![1.0, 0.0, 0.0]),
            (2, vec![0.0, 1.0, 0.0]),
            (3, vec![0.2, 0.8, 0.0]),
        ];
        let idx = build_from_dataset(&data, params);
        let q = vec![1.0, 0.0, 0.0];
        let before = idx.search(&q, 2);

        let mut buf = Vec::new();
        idx.save(&mut buf).unwrap();
        let loaded = TrieIndex::load(&mut buf.as_slice()).unwrap();
        let after = loaded.search(&q, 2);
        assert_eq!(before, after);
    }

    #[test]
    fn bad_version_errors() {
        let mut buf = Vec::new();
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&99u32.to_le_bytes());
        let err = TrieIndex::load(&mut buf.as_slice()).unwrap_err();
        match err {
            SerdeError::UnsupportedVersion(99) => {}
            other => panic!("unexpected {other}"),
        }
    }
}
