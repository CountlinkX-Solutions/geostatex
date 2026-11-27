//! numpy's legacy random draws, bit for bit: `RandomState(MT19937(SeedSequence(seed)))` and its
//! `choice(n, k, replace=False)`, which scikit-gstat uses to sample point pairs. The algorithms
//! are numpy's published ones (SeedSequence after O'Neill's `seed_seq`, the Mersenne Twister,
//! the legacy masked bounded integers and Fisher-Yates shuffle).

const INIT_A: u32 = 0x43b0_d7e5;
const MULT_A: u32 = 0x931e_8875;
const INIT_B: u32 = 0x8b51_f9dd;
const MULT_B: u32 = 0x58f3_8ded;
const MIX_MULT_L: u32 = 0xca01_f9dd;
const MIX_MULT_R: u32 = 0x4973_f715;
const XSHIFT: u32 = 16;
const POOL_SIZE: usize = 4;

/// `SeedSequence(seed).generate_state(n, uint32)`.
pub fn seed_words(seed: u64, n: usize) -> Vec<u32> {
    // The integer as little-endian 32-bit words (one word for 0).
    let entropy: Vec<u32> = if seed >> 32 == 0 {
        vec![seed as u32]
    } else {
        vec![seed as u32, (seed >> 32) as u32]
    };
    let mut hash_const = INIT_A;
    let mut hashmix = |value: u32| {
        let mut v = value ^ hash_const;
        hash_const = hash_const.wrapping_mul(MULT_A);
        v = v.wrapping_mul(hash_const);
        v ^ (v >> XSHIFT)
    };
    let mix = |x: u32, y: u32| {
        let r = MIX_MULT_L
            .wrapping_mul(x)
            .wrapping_sub(MIX_MULT_R.wrapping_mul(y));
        r ^ (r >> XSHIFT)
    };
    let mut pool: Vec<u32> = (0..POOL_SIZE)
        .map(|i| hashmix(entropy.get(i).copied().unwrap_or(0)))
        .collect();
    for src in 0..POOL_SIZE {
        for dst in 0..POOL_SIZE {
            if src != dst {
                pool[dst] = mix(pool[dst], hashmix(pool[src]));
            }
        }
    }
    // Entropy beyond the pool (never for a u64 seed: at most 2 words).
    for &word in entropy.iter().skip(POOL_SIZE) {
        for slot in pool.iter_mut() {
            *slot = mix(*slot, hashmix(word));
        }
    }
    let mut hash_const = INIT_B;
    (0..n)
        .map(|i| {
            let mut v = pool[i % POOL_SIZE] ^ hash_const;
            hash_const = hash_const.wrapping_mul(MULT_B);
            v = v.wrapping_mul(hash_const);
            v ^ (v >> XSHIFT)
        })
        .collect()
}

const N: usize = 624;
const M: usize = 397;

/// numpy's `MT19937` bit generator seeded through a `SeedSequence`.
pub struct Mt19937 {
    key: [u32; N],
    pos: usize,
}

impl Mt19937 {
    pub fn new(seed: u64) -> Self {
        let words = seed_words(seed, N);
        let mut key = [0u32; N];
        key.copy_from_slice(&words);
        // numpy sets the most significant bit of the first word (a non-zero state) and leaves
        // the position at 623, so the first output tempers key[623] before any regeneration.
        key[0] = 0x8000_0000;
        Mt19937 { key, pos: N - 1 }
    }

    /// numpy's legacy integer seeding (`RandomState(seed)`: `mt19937_seed`, `init_genrand`).
    pub fn legacy(seed: u32) -> Self {
        let mut key = [0u32; N];
        key[0] = seed;
        for i in 1..N {
            key[i] = 1_812_433_253_u32
                .wrapping_mul(key[i - 1] ^ (key[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        Mt19937 { key, pos: N }
    }

    /// A double in `[0, 1)` from two outputs (53 bits), numpy's `mt19937_next_double`.
    pub fn next_double(&mut self) -> f64 {
        let a = self.next_u32() >> 5;
        let b = self.next_u32() >> 6;
        (a as f64 * 67_108_864.0 + b as f64) / 9_007_199_254_740_992.0
    }

    fn regenerate(&mut self) {
        let k = &mut self.key;
        for i in 0..N {
            let y = (k[i] & 0x8000_0000) | (k[(i + 1) % N] & 0x7fff_ffff);
            let mag = if y & 1 == 1 { 0x9908_b0df } else { 0 };
            k[i] = k[(i + M) % N] ^ (y >> 1) ^ mag;
        }
        self.pos = 0;
    }

    pub fn next_u32(&mut self) -> u32 {
        if self.pos >= N {
            self.regenerate();
        }
        let mut y = self.key[self.pos];
        self.pos += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    /// numpy's legacy `random_interval`: a uniform integer in `[0, max]` by masked rejection.
    pub(crate) fn interval(&mut self, max: u32) -> u32 {
        if max == 0 {
            return 0;
        }
        let mut mask = max;
        for shift in [1, 2, 4, 8, 16] {
            mask |= mask >> shift;
        }
        loop {
            let v = self.next_u32() & mask;
            if v <= max {
                return v;
            }
        }
    }
}

/// `RandomState.choice(n, k, replace=False)`: the first `k` of a legacy permutation of `0..n`.
pub fn choice(mt: &mut Mt19937, n: usize, k: usize) -> Vec<usize> {
    let mut arr: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = mt.interval(i as u32) as usize;
        arr.swap(i, j);
    }
    arr.truncate(k);
    arr
}

#[cfg(test)]
mod tests {
    use super::*;

    // Reference values recorded from numpy 2.5 (test/reference/fixtures/variogram/samples.json).

    #[test]
    fn seed_sequence_words() {
        assert_eq!(
            seed_words(0, 4),
            vec![2968811710, 3677149159, 745650761, 2884920346]
        );
        assert_eq!(
            seed_words(u64::MAX, 4),
            vec![2458692877, 2931597649, 2251873402, 295448644]
        );
    }

    #[test]
    fn mt19937_outputs_across_the_first_regeneration() {
        let mut mt = Mt19937::new(42);
        let out: Vec<u32> = (0..626).map(|_| mt.next_u32()).collect();
        assert_eq!(out[..3], [2327846034, 3904886566, 2661450408]);
        assert_eq!(out[623..626], [300529090, 1432877092, 4269216375]);
    }

    #[test]
    fn two_successive_choices() {
        let mut mt = Mt19937::new(1);
        assert_eq!(choice(&mut mt, 10, 3), vec![1, 0, 9]);
        assert_eq!(choice(&mut mt, 10, 3), vec![6, 8, 5]);
        let mut mt = Mt19937::new(u64::MAX);
        assert_eq!(choice(&mut mt, 10, 3), vec![5, 9, 4]);
        assert_eq!(choice(&mut mt, 10, 3), vec![0, 2, 6]);
        let mut mt = Mt19937::new((1 << 63) + 17);
        assert_eq!(
            choice(&mut mt, 60, 60)[..8],
            [30, 18, 37, 43, 14, 25, 17, 24]
        );
        assert_eq!(choice(&mut mt, 60, 60)[..8], [17, 36, 6, 5, 34, 45, 57, 46]);
    }
}
