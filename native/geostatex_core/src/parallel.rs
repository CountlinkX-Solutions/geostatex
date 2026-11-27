//! Deterministic parallel map over indices with scoped threads.

use std::sync::atomic::{AtomicUsize, Ordering};

/// `f(0), f(1), ..., f(n - 1)` in order, computed on up to the available cores, each thread
/// taking a contiguous run of at least `min_per_thread` indices. Every value is computed exactly
/// as a sequential map computes it.
pub fn par_map<T: Send, F: Fn(usize) -> T + Sync>(n: usize, min_per_thread: usize, f: F) -> Vec<T> {
    let cores = std::thread::available_parallelism().map_or(1, |c| c.get());
    let threads = cores.min(n / min_per_thread.max(1)).max(1);
    if threads == 1 {
        return (0..n).map(f).collect();
    }
    let chunk = n.div_ceil(threads);
    std::thread::scope(|scope| {
        let f = &f;
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                scope.spawn(move || {
                    (t * chunk..((t + 1) * chunk).min(n))
                        .map(f)
                        .collect::<Vec<T>>()
                })
            })
            .collect();
        let mut out = Vec::with_capacity(n);
        for h in handles {
            out.extend(h.join().unwrap_or_else(|e| std::panic::resume_unwind(e)));
        }
        out
    })
}

/// `f(0), f(1), ..., f(n - 1)` in order, for work whose cost varies a lot from one index to
/// another: up to the available cores (and at most `max_threads`) each take the next index not
/// yet taken, one at a time. Every value is computed exactly as a sequential map computes it.
pub fn par_map_dynamic<T: Send, F: Fn(usize) -> T + Sync>(
    n: usize,
    max_threads: usize,
    f: F,
) -> Vec<T> {
    let cores = std::thread::available_parallelism().map_or(1, |c| c.get());
    let threads = cores.min(n).min(max_threads);
    if threads <= 1 {
        return (0..n).map(f).collect();
    }
    let next = AtomicUsize::new(0);
    let mut out: Vec<Option<T>> = (0..n).map(|_| None).collect();
    std::thread::scope(|scope| {
        let (f, next) = (&f, &next);
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                scope.spawn(move || {
                    let mut done = Vec::new();
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= n {
                            return done;
                        }
                        done.push((i, f(i)));
                    }
                })
            })
            .collect();
        for h in handles {
            for (i, v) in h.join().unwrap_or_else(|e| std::panic::resume_unwind(e)) {
                out[i] = Some(v);
            }
        }
    });
    out.into_iter()
        .map(|v| v.expect("every index is taken once"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn par_map_equals_a_sequential_map() {
        let f = |i: usize| ((i as f64).sqrt() * 1.000_000_1).sin().to_bits();
        for n in [0, 1, 2, 7, 64, 1000, 4097] {
            for min in [1, 3, 100] {
                let seq: Vec<u64> = (0..n).map(f).collect();
                assert_eq!(par_map(n, min, f), seq, "n {n}, min {min}");
            }
        }
    }

    #[test]
    fn par_map_dynamic_equals_a_sequential_map() {
        // Uneven work: index i costs about i steps, as the statistics' systems grow with i.
        let f = |i: usize| {
            (0..i * 50).fold((i as f64).sqrt(), |acc, j| {
                (acc + j as f64 * 1e-3).sin() + 1.0
            })
        };
        for n in [0, 1, 2, 7, 64, 333] {
            let seq: Vec<u64> = (0..n).map(|i| f(i).to_bits()).collect();
            for threads in [1, 3, usize::MAX] {
                let par: Vec<u64> = par_map_dynamic(n, threads, f)
                    .iter()
                    .map(|v| v.to_bits())
                    .collect();
                assert_eq!(par, seq, "n {n}, threads {threads}");
            }
        }
    }
}
