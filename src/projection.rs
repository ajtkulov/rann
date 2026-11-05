use rand::Rng;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::types::BitCode;

/// Random-hyperplane LSH projector.
#[derive(Debug, Clone)]
pub struct Projector {
    pub dim: usize,
    pub h: usize,
    pub seed: u64,
    /// `h` planes, each of length `dim`.
    pub planes: Vec<Vec<f32>>,
}

impl Projector {
    pub fn new(dim: usize, h: usize, seed: u64) -> Self {
        assert!(dim > 0 && h > 0);
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut planes = Vec::with_capacity(h);
        for _ in 0..h {
            let mut plane = Vec::with_capacity(dim);
            for _ in 0..dim {
                // Box-Muller-ish via standard normal approximation from rand
                let u1: f32 = rng.gen::<f32>().max(f32::EPSILON);
                let u2: f32 = rng.gen();
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
                plane.push(z);
            }
            planes.push(plane);
        }
        Self {
            dim,
            h,
            seed,
            planes,
        }
    }

    pub fn from_planes(dim: usize, h: usize, seed: u64, planes: Vec<Vec<f32>>) -> Self {
        assert_eq!(planes.len(), h);
        for p in &planes {
            assert_eq!(p.len(), dim);
        }
        Self {
            dim,
            h,
            seed,
            planes,
        }
    }

    pub fn project(&self, vector: &[f32]) -> BitCode {
        assert_eq!(vector.len(), self.dim);
        let mut code = BitCode::zeros(self.h);
        for (i, plane) in self.planes.iter().enumerate() {
            let mut dot = 0.0f32;
            for (a, b) in vector.iter().zip(plane.iter()) {
                dot += a * b;
            }
            code.set_bit(i, dot >= 0.0);
        }
        code
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_h_is_132() {
        let p = Projector::new(8, 132, 1729);
        assert_eq!(p.h, 132);
        assert_eq!(p.project(&vec![0.0; 8]).bit_len(), 132);
    }

    #[test]
    fn same_seed_identical_codes() {
        let a = Projector::new(4, 16, 1729);
        let b = Projector::new(4, 16, 1729);
        let v = [1.0, -2.0, 0.5, 3.0];
        assert_eq!(a.planes, b.planes);
        assert_eq!(a.project(&v), b.project(&v));
    }

    #[test]
    fn bit_polarity_half_space() {
        // One plane = e0, so sign depends on x[0]
        let planes = vec![vec![1.0, 0.0]];
        let p = Projector::from_planes(2, 1, 0, planes);
        assert!(p.project(&[1.0, 0.0]).get_bit(0));
        assert!(!p.project(&[-1.0, 0.0]).get_bit(0));
        assert!(p.project(&[0.0, 5.0]).get_bit(0)); // dot == 0 => bit 1
    }

    #[test]
    fn planes_roundtrip_construct() {
        let p = Projector::new(3, 5, 42);
        let q = Projector::from_planes(3, 5, 42, p.planes.clone());
        let v = [0.1, 0.2, -0.3];
        assert_eq!(p.project(&v), q.project(&v));
    }
}
