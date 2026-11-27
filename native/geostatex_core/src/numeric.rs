//! The numpy routines the reference libraries rely on, reproduced exactly where the result
//! depends on them: `linspace`, the `linear` percentile, `median`, `std` and `histogram` counts.

/// `numpy.linspace(start, stop, num)` with `endpoint=True`: `start + i * step`, the last value
/// set to `stop` exactly.
pub fn linspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
    if num == 0 {
        return Vec::new();
    }
    if num == 1 {
        return vec![start];
    }
    let div = (num - 1) as f64;
    let step = (stop - start) / div;
    let mut out: Vec<f64> = (0..num).map(|i| start + i as f64 * step).collect();
    out[num - 1] = stop;
    out
}

/// numpy's `_lerp`: `a + (b - a) * t`, computed from `b` when `t >= 0.5`.
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    let diff = b - a;
    if t >= 0.5 {
        b - diff * (1.0 - t)
    } else {
        a + diff * t
    }
}

/// `numpy.quantile(x, q)` with the default `linear` method on already sorted, NaN-free data.
/// `None` for empty input.
pub fn quantile_sorted(sorted: &[f64], q: f64) -> Option<f64> {
    let n = sorted.len();
    if n == 0 {
        return None;
    }
    let virtual_index = q * (n - 1) as f64;
    let lo = virtual_index.floor();
    let gamma = virtual_index - lo;
    let lo = (lo as usize).min(n - 1);
    let hi = (lo + 1).min(n - 1);
    Some(lerp(sorted[lo], sorted[hi], gamma))
}

/// `numpy.percentile(x, p)` (linear) of unsorted data, NaN dropped (`nanpercentile`).
pub fn nanpercentile(values: &[f64], p: f64) -> Option<f64> {
    let mut sorted: Vec<f64> = values.iter().copied().filter(|v| !v.is_nan()).collect();
    sorted.sort_by(|a, b| a.total_cmp(b));
    quantile_sorted(&sorted, p / 100.0)
}

/// `numpy.median` (mean of the two middle values for an even count), NaN dropped.
pub fn nanmedian(values: &[f64]) -> Option<f64> {
    let mut sorted: Vec<f64> = values.iter().copied().filter(|v| !v.is_nan()).collect();
    if sorted.is_empty() {
        return None;
    }
    sorted.sort_by(|a, b| a.total_cmp(b));
    let n = sorted.len();
    Some(if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    })
}

/// Arithmetic mean (`None` for empty input).
pub fn mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        None
    } else {
        Some(values.iter().sum::<f64>() / values.len() as f64)
    }
}

/// Population standard deviation, `numpy.std(x)` (`ddof = 0`).
pub fn std(values: &[f64]) -> Option<f64> {
    let m = mean(values)?;
    let var = values.iter().map(|v| (v - m).abs().powi(2)).sum::<f64>() / values.len() as f64;
    Some(var.sqrt())
}

/// Counts of `numpy.histogram(x, bins=edges)`: bin `i` holds `edges[i] <= x < edges[i + 1]`, the
/// last bin also `x == edges[-1]`; values outside the edges (and NaN) are dropped.
pub fn histogram_counts(values: &[f64], edges: &[f64]) -> Vec<usize> {
    let bins = edges.len().saturating_sub(1);
    let mut counts = vec![0usize; bins];
    if bins == 0 {
        return counts;
    }
    let (first, last) = (edges[0], edges[bins]);
    for &x in values {
        if x.is_nan() || x < first || x > last {
            continue;
        }
        let i = if x == last {
            bins - 1
        } else {
            edges.partition_point(|&e| e <= x) - 1
        };
        counts[i] += 1;
    }
    counts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linspace_matches_numpy() {
        assert_eq!(linspace(0.0, 1.0, 5), vec![0.0, 0.25, 0.5, 0.75, 1.0]);
        assert_eq!(linspace(2.0, 3.0, 1), vec![2.0]);
        assert_eq!(*linspace(0.0, 0.3, 4).last().unwrap(), 0.3);
    }

    #[test]
    fn linear_percentile_and_median() {
        let x = [3.0, 1.0, 4.0, 1.0, 5.0];
        assert_eq!(nanpercentile(&x, 50.0), Some(3.0));
        assert_eq!(nanpercentile(&x, 25.0), Some(1.0));
        assert_eq!(nanpercentile(&[1.0, 2.0], 75.0), Some(1.75));
        assert_eq!(nanmedian(&[4.0, 1.0, 3.0, 2.0]), Some(2.5));
        assert_eq!(nanmedian(&[f64::NAN]), None);
        assert_eq!(nanpercentile(&[], 50.0), None);
    }

    #[test]
    fn std_is_the_population_one() {
        let s = std(&[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]).unwrap();
        assert!((s - 2.0).abs() < 1e-15);
    }

    #[test]
    fn histogram_closes_the_last_bin_and_drops_outliers() {
        let edges = [0.0, 1.0, 2.0, 3.0];
        let x = [-1.0, 0.0, 0.5, 1.0, 2.999, 3.0, 3.5, f64::NAN];
        assert_eq!(histogram_counts(&x, &edges), vec![2, 1, 2]);
    }
}
