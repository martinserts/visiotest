//! A tiny pseudo-random generator for the synthetic scenes.
//!
//! No `rand` crate: the scenes need a handful of noisy numbers, the same ones on every
//! run, and a dozen lines of xorshift64* do that. It is *not* for anything that needs
//! real randomness (keys, shuffling secrets): it only has to look noisy and be
//! reproducible.

/// xorshift64* (Vigna, 2014): three shifts scramble a 64-bit state, and one multiply
/// spreads the bits of the result.
pub struct Rng {
    state: u64,
}

impl Rng {
    /// A generator seeded with `seed`. Equal seeds give equal sequences.
    pub fn new(seed: u64) -> Self {
        // An all-zero state would stay zero forever. XOR with a constant (the 64-bit
        // golden ratio) keeps small seeds like 1, 2, 3 from that, and spreads their bits.
        let state = seed ^ 0x9E37_79B9_7F4A_7C15;
        Self {
            state: if state == 0 { 1 } else { state },
        }
    }

    /// The next raw 64-bit number.
    pub fn next_u64(&mut self) -> u64 {
        self.state ^= self.state >> 12;
        self.state ^= self.state << 25;
        self.state ^= self.state >> 27;
        self.state.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in `[0, 1)`.
    pub fn next_f32(&mut self) -> f32 {
        // The top 24 bits fill an f32's 24-bit mantissa exactly: every value is
        // k / 2²⁴, so 1.0 itself can never come out.
        (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
    }

    /// Uniform in `[lo, hi)`.
    pub fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next_f32()
    }

    /// Standard normal (mean 0, std 1), by the Box–Muller transform: two uniforms
    /// `u1, u2` give `√(−2 ln u1) · cos(2π u2)`.
    pub fn normal(&mut self) -> f32 {
        // `1 − u` is in (0, 1], so the logarithm never sees 0.
        let u1 = 1.0 - self.next_f32();
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sim_rng_is_reproducible_and_in_range() {
        let (mut a, mut b) = (Rng::new(7), Rng::new(7));
        for _ in 0..1000 {
            let x = a.next_f32();
            assert_eq!(x, b.next_f32());
            assert!((0.0..1.0).contains(&x));
        }
        assert_ne!(Rng::new(1).next_u64(), Rng::new(2).next_u64());
    }

    #[test]
    fn sim_rng_normal_has_unit_spread() {
        let mut rng = Rng::new(42);
        let n = 20_000;
        let samples: Vec<f32> = (0..n).map(|_| rng.normal()).collect();
        let mean = samples.iter().sum::<f32>() / n as f32;
        let var = samples.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n as f32;
        assert!(mean.abs() < 0.03, "mean {mean}");
        assert!((var - 1.0).abs() < 0.05, "variance {var}");
    }
}
