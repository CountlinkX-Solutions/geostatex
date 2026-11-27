//! Experimental variograms after `skgstat.Variogram`: pairwise distances and value differences in
//! `scipy.spatial.distance.pdist` order (`i < j`), the lag edges of scikit-gstat's binning
//! functions, `numpy.digitize` grouping and the semivariance estimators.

use crate::memory;
use crate::numeric::{
    histogram_counts, linspace, mean, nanmedian, nanpercentile, quantile_sorted, std,
};

/// Distance metric between coordinates (the `scipy` names, with scipy's defaults: `SEuclidean`
/// divides by the column variances and `Mahalanobis` uses the inverse covariance of the
/// coordinates; `Minkowski` has p = 2, which scipy computes as the Euclidean distance).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Metric {
    Euclidean,
    Cityblock,
    Chebyshev,
    SqEuclidean,
    SEuclidean,
    Canberra,
    Braycurtis,
    Cosine,
    Mahalanobis,
    Minkowski,
}

/// How the largest lag is set (`Variogram.maxlag`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MaxLag {
    /// No limit: the edges reach the largest distance.
    Unset,
    /// A share of the largest distance when below 1, an absolute distance otherwise.
    Value(f64),
    Mean,
    Median,
}

/// numpy `histogram_bin_edges` rules used by scikit-gstat's `auto_derived_lags`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Rule {
    Sturges,
    Scott,
    Sqrt,
    Fd,
    Doane,
    Rice,
}

/// The lag binning (`bin_func`).
#[derive(Clone, Debug, PartialEq)]
pub enum Bins {
    /// `n_lags` equal-width lags from 0 to `maxlag`.
    Even,
    /// `n_lags` lags holding the same number of pairs.
    Uniform,
    /// The number of lags follows from a numpy rule.
    Rule(Rule),
    /// Explicit upper edges.
    Edges(Vec<f64>),
}

/// Semivariance estimator of one lag class.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Estimator {
    Matheron,
    Cressie,
    Dowd,
    Genton,
    Minmax,
    Percentile(f64),
    /// Shannon entropy over `entropy_bins - 1` equal bins of the distance range.
    Entropy(usize),
}

/// How far a point pair may lie across the azimuth (`DirectionalVariogram.bandwidth`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Bandwidth {
    /// A coordinate distance; one beyond the largest pair distance has no effect.
    Value(f64),
    /// A percentile of the pair distances (scikit-gstat's `"q33"`).
    Percentile(f64),
}

/// The search area of a directional variogram (`directional_model`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SearchArea {
    /// Within the tolerance angle and the bandwidth across the azimuth.
    Triangle,
    /// Within the tolerance angle; the bandwidth is ignored.
    Compass,
}

/// The direction of a `skgstat.DirectionalVariogram`: azimuth and tolerance in degrees (the
/// azimuth from east, clockwise positive).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Direction {
    pub azimuth: f64,
    pub tolerance: f64,
    pub bandwidth: Bandwidth,
    pub area: SearchArea,
}

/// How an experimental variogram is computed (`skgstat.Variogram` arguments).
#[derive(Clone, Debug, PartialEq)]
pub struct Options {
    pub metric: Metric,
    pub maxlag: MaxLag,
    pub bins: Bins,
    pub n_lags: usize,
    pub estimator: Estimator,
    /// Only the point pairs in this direction (`skgstat.DirectionalVariogram`).
    pub direction: Option<Direction>,
    /// A cross-variogram with this co-variable (one value per point).
    pub cross: Option<Cross>,
    /// Only the pairs between two random subsets of the points (`samples`).
    pub sampling: Option<Sampling>,
    /// Also compute scikit-gstat's entropy fit sigma (`fit_sigma="entropy"`).
    pub entropy_sigma: bool,
}

/// scikit-gstat's pair sampling (`ProbabalisticMetricSpace`): two subsets of `k` points drawn
/// with numpy's `RandomState(MT19937(SeedSequence(seed)))`, the left one first; the pairs are
/// `(l, r)` with `r < l` and a non-zero distance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sampling {
    pub k: usize,
    pub seed: u64,
}

/// The two cross-variogram definitions, each holding the co-variable.
#[derive(Clone, Debug, PartialEq)]
pub enum Cross {
    /// scikit-gstat's: the estimator runs on `|a_i - a_j| * |b_i - b_j|`.
    Skgstat(Vec<f64>),
    /// The classical cross-semivariogram `sum((a_i - a_j) (b_i - b_j)) / 2N` (Matheron only).
    Classical(Vec<f64>),
}

/// An experimental variogram.
#[derive(Clone, Debug, PartialEq)]
pub struct Experimental {
    /// Upper edges of the lag classes.
    pub bins: Vec<f64>,
    /// Semivariance per lag (NaN for an empty lag).
    pub experimental: Vec<f64>,
    /// Number of point pairs per lag.
    pub bin_count: Vec<usize>,
    /// The resolved `maxlag` (`None` when unset).
    pub maxlag: Option<f64>,
    /// The resolved bandwidth of a directional variogram.
    pub bandwidth: Option<f64>,
    /// `1 / h` per lag (NaN for an empty one), `h` the Shannon entropy of the lag's value
    /// differences over Scott's edges of all the pair distances, when requested.
    pub entropy_sigma: Option<Vec<f64>>,
}

fn pair_count(coords: &[f64], dim: usize) -> usize {
    let n = coords.len() / dim;
    n * n.saturating_sub(1) / 2
}

/// Column `k` of row-major coordinates.
fn column(coords: &[f64], dim: usize, k: usize) -> Vec<f64> {
    coords.iter().skip(k).step_by(dim).copied().collect()
}

/// numpy's `var(ddof=1)` of every coordinate column (`seuclidean`'s default `V`).
fn column_variances(coords: &[f64], dim: usize) -> Vec<f64> {
    (0..dim)
        .map(|k| {
            let c = column(coords, dim, k);
            let m = c.iter().sum::<f64>() / c.len() as f64;
            c.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (c.len() as f64 - 1.0)
        })
        .collect()
}

/// `inv(cov(X.T)).T` (`mahalanobis`' default `VI`), row-major.
fn inverse_covariance(coords: &[f64], dim: usize) -> Result<Vec<f64>, String> {
    let cols: Vec<Vec<f64>> = (0..dim).map(|k| column(coords, dim, k)).collect();
    let n = cols[0].len() as f64;
    let centered: Vec<Vec<f64>> = cols
        .iter()
        .map(|c| {
            let m = c.iter().sum::<f64>() / n;
            c.iter().map(|x| x - m).collect()
        })
        .collect();
    let cov: Vec<Vec<f64>> = (0..dim)
        .map(|i| {
            (0..dim)
                .map(|j| {
                    centered[i]
                        .iter()
                        .zip(&centered[j])
                        .map(|(a, b)| a * b)
                        .sum::<f64>()
                        / (n - 1.0)
                })
                .collect()
        })
        .collect();
    let mut vi = Vec::with_capacity(dim * dim);
    // Row i of inv(C).T is column i of inv(C): the solution of C x = e_i.
    for i in 0..dim {
        let mut e = vec![0.0; dim];
        e[i] = 1.0;
        let x = crate::lsq::solve_small(cov.clone(), e)
            .filter(|x| x.iter().all(|v| v.is_finite()))
            .ok_or("mahalanobis needs coordinates with a non-singular covariance")?;
        vi.extend(x);
    }
    Ok(vi)
}

/// Condensed pairwise distances of `n` points of dimension `dim` (row-major coordinates), or why
/// a metric cannot measure them (fewer than 2 points for `seuclidean` and `mahalanobis`, or a
/// singular covariance).
pub fn pdist(coords: &[f64], dim: usize, metric: Metric) -> Result<Vec<f64>, String> {
    let n = coords.len() / dim;
    if matches!(metric, Metric::SEuclidean | Metric::Mahalanobis) && n < 2 {
        return Err(format!("{metric:?} needs at least 2 points"));
    }
    let variances = match metric {
        Metric::SEuclidean => column_variances(coords, dim),
        _ => Vec::new(),
    };
    let vi = match metric {
        Metric::Mahalanobis => inverse_covariance(coords, dim)?,
        _ => Vec::new(),
    };
    let norms: Vec<f64> = match metric {
        Metric::Cosine => (0..n)
            .map(|i| {
                let a = &coords[i * dim..(i + 1) * dim];
                a.iter().map(|x| x * x).sum::<f64>().sqrt()
            })
            .collect(),
        _ => Vec::new(),
    };
    let mut out = Vec::with_capacity(pair_count(coords, dim));
    for i in 0..n {
        let a = &coords[i * dim..(i + 1) * dim];
        for j in (i + 1)..n {
            let b = &coords[j * dim..(j + 1) * dim];
            let pairs = || a.iter().zip(b);
            // The operation order follows scipy's C implementations.
            let d = match metric {
                Metric::Euclidean | Metric::Minkowski => {
                    pairs().map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt()
                }
                Metric::Cityblock => pairs().map(|(x, y)| (x - y).abs()).sum(),
                Metric::Chebyshev => pairs().map(|(x, y)| (x - y).abs()).fold(0.0, f64::max),
                Metric::SqEuclidean => pairs().map(|(x, y)| (x - y) * (x - y)).sum(),
                Metric::SEuclidean => pairs()
                    .zip(&variances)
                    .map(|((x, y), v)| (x - y) * (x - y) / v)
                    .sum::<f64>()
                    .sqrt(),
                Metric::Canberra => pairs()
                    .map(|(x, y)| {
                        let denominator = x.abs() + y.abs();
                        if denominator > 0.0 {
                            (x - y).abs() / denominator
                        } else {
                            0.0
                        }
                    })
                    .sum(),
                Metric::Braycurtis => {
                    let num: f64 = pairs().map(|(x, y)| (x - y).abs()).sum();
                    let den: f64 = pairs().map(|(x, y)| (x + y).abs()).sum();
                    num / den
                }
                Metric::Cosine => {
                    let dot: f64 = pairs().map(|(x, y)| x * y).sum();
                    let cosine = dot / (norms[i] * norms[j]);
                    1.0 - if cosine.abs() > 1.0 {
                        cosine.signum()
                    } else {
                        cosine
                    }
                }
                Metric::Mahalanobis => {
                    let delta: Vec<f64> = pairs().map(|(x, y)| x - y).collect();
                    let projected: Vec<f64> = (0..dim)
                        .map(|r| (0..dim).map(|c| delta[c] * vi[r * dim + c]).sum())
                        .collect();
                    delta
                        .iter()
                        .zip(&projected)
                        .map(|(d, p)| d * p)
                        .sum::<f64>()
                        .sqrt()
                }
            };
            out.push(d);
        }
    }
    Ok(out)
}

/// Absolute value differences `|z_i - z_j|` in `pdist` order.
pub fn pairwise_diffs(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut out = Vec::with_capacity(n * n.saturating_sub(1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            out.push((values[i] - values[j]).abs());
        }
    }
    out
}

/// The sampled pairs `(l, r, distance)`, in scikit-gstat's (CSR) order: by `l`, then `r`.
fn sampled_pairs(
    coords: &[f64],
    dim: usize,
    s: Sampling,
) -> Result<Vec<(usize, usize, f64)>, String> {
    let n = coords.len() / dim;
    if s.k == 0 || s.k > n {
        return Err(format!(
            "samples must select between 1 and {n} points, got {}",
            s.k
        ));
    }
    if u32::try_from(n).is_err() {
        return Err("pair sampling supports at most 2^32 - 1 points".to_string());
    }
    let mut mt = crate::numpy_random::Mt19937::new(s.seed);
    let left = crate::numpy_random::choice(&mut mt, n, s.k);
    let right = crate::numpy_random::choice(&mut mt, n, s.k);
    let mut pairs: Vec<(usize, usize)> = left
        .iter()
        .flat_map(|&l| right.iter().filter(move |&&r| r < l).map(move |&r| (l, r)))
        .collect();
    pairs.sort_unstable();
    Ok(pairs
        .into_iter()
        .filter_map(|(l, r)| {
            let (a, b) = (
                &coords[l * dim..(l + 1) * dim],
                &coords[r * dim..(r + 1) * dim],
            );
            let d = a
                .iter()
                .zip(b)
                .map(|(x, y)| (x - y) * (x - y))
                .sum::<f64>()
                .sqrt();
            // The sparse product scikit-gstat filters with drops the zero distances.
            (d > 0.0).then_some((l, r, d))
        })
        .collect())
}

fn max_of(values: &[f64]) -> f64 {
    values.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}

/// The `maxlag` scikit-gstat stores for the distances.
pub fn resolve_maxlag(maxlag: MaxLag, distances: &[f64]) -> Option<f64> {
    match maxlag {
        MaxLag::Unset => None,
        MaxLag::Value(v) if v < 1.0 => Some(v * max_of(distances)),
        MaxLag::Value(v) => Some(v),
        MaxLag::Mean => mean(distances),
        MaxLag::Median => nanmedian(distances),
    }
}

/// `maxlag` capped at the largest distance, as every binning function does first.
fn cap(maxlag: Option<f64>, distances: &[f64]) -> f64 {
    let largest = max_of(distances);
    match maxlag {
        Some(m) if m <= largest => m,
        _ => largest,
    }
}

/// numpy's outer edges of a histogram range (`_get_outer_edges`).
fn outer_edges(values: &[f64]) -> (f64, f64) {
    if values.is_empty() {
        return (0.0, 1.0);
    }
    let first = values.iter().copied().fold(f64::INFINITY, f64::min);
    let last = max_of(values);
    if first == last {
        (first - 0.5, last + 0.5)
    } else {
        (first, last)
    }
}

/// Bin width of a numpy rule (`_hist_bin_*`); 0 means "one bin".
fn rule_width(rule: Rule, x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let ptp = max_of(x) - x.iter().copied().fold(f64::INFINITY, f64::min);
    match rule {
        Rule::Sqrt => ptp / n.sqrt(),
        Rule::Sturges => ptp / (n.log2() + 1.0),
        Rule::Rice => ptp / (2.0 * n.powf(1.0 / 3.0)),
        Rule::Scott => {
            (24.0 * std::f64::consts::PI.sqrt() / n).powf(1.0 / 3.0) * std(x).unwrap_or(0.0)
        }
        Rule::Fd => {
            let iqr = nanpercentile(x, 75.0).unwrap_or(0.0) - nanpercentile(x, 25.0).unwrap_or(0.0);
            2.0 * iqr * n.powf(-1.0 / 3.0)
        }
        Rule::Doane => {
            if x.len() <= 2 {
                return 0.0;
            }
            let sg1 = (6.0 * (n - 2.0) / ((n + 1.0) * (n + 3.0))).sqrt();
            let sigma = std(x).unwrap_or(0.0);
            if sigma <= 0.0 {
                return 0.0;
            }
            let m = mean(x).unwrap_or(0.0);
            let g1 = x.iter().map(|v| ((v - m) / sigma).powi(3)).sum::<f64>() / n;
            ptp / (1.0 + n.log2() + (1.0 + g1.abs() / sg1).log2())
        }
    }
}

/// `numpy.histogram_bin_edges(x, bins=rule)`.
fn rule_edges(rule: Rule, x: &[f64]) -> Result<Vec<f64>, String> {
    let (first, last) = outer_edges(x);
    let n = if x.is_empty() {
        1
    } else {
        let width = rule_width(rule, x);
        if width != 0.0 && !width.is_nan() {
            ((last - first) / width).ceil() as usize
        } else {
            1
        }
    };
    let edges = linspace(first, last, n + 1);
    if edges.windows(2).any(|w| w[0] >= w[1]) {
        return Err(format!(
            "too many lags for the distance range: cannot create {n} finite-sized lags"
        ));
    }
    Ok(edges)
}

/// Upper lag edges for `bins` (scikit-gstat's binning functions).
pub fn lag_edges(
    bins: &Bins,
    distances: &[f64],
    n_lags: usize,
    maxlag: Option<f64>,
) -> Result<Vec<f64>, String> {
    match bins {
        Bins::Edges(edges) => Ok(edges.clone()),
        Bins::Even => Ok(linspace(0.0, cap(maxlag, distances), n_lags + 1)[1..].to_vec()),
        Bins::Uniform => {
            let m = cap(maxlag, distances);
            let kept: Vec<f64> = distances.iter().copied().filter(|d| *d <= m).collect();
            (1..=n_lags)
                .map(|i| {
                    nanpercentile(&kept, i as f64 / n_lags as f64 * 100.0)
                        .ok_or_else(|| "no distance within maxlag".to_string())
                })
                .collect()
        }
        Bins::Rule(rule) => {
            let m = cap(maxlag, distances);
            let kept: Vec<f64> = distances.iter().copied().filter(|d| *d <= m).collect();
            Ok(rule_edges(*rule, &kept)?[1..].to_vec())
        }
    }
}

/// `numpy.digitize(d, edges)` (`right=False`): the number of edges `<= d`; `None` for the group
/// beyond the last edge, which scikit-gstat leaves out.
fn group_of(d: f64, edges: &[f64]) -> Option<usize> {
    let g = edges.partition_point(|&e| e <= d);
    (g < edges.len()).then_some(g)
}

/// `scipy.special.binom(x, 2)` for a real `x`, in scipy's product form.
fn binom2(x: f64) -> f64 {
    let kx = if x.fract() == 0.0 && x > 0.0 && 2.0 > x / 2.0 {
        x - 2.0
    } else {
        2.0
    };
    if kx <= 0.0 {
        return 1.0;
    }
    let mut num = 1.0;
    let mut den = 1.0;
    let mut i = 1.0;
    while i <= kx {
        num *= i + x - kx;
        den *= i;
        i += 1.0;
    }
    num / den
}

fn genton(x: &[f64]) -> f64 {
    let n = x.len();
    if n < 2 {
        return f64::NAN;
    }
    let mut y = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            y.push((x[i] - x[j]).abs());
        }
    }
    y.sort_by(|a, b| a.total_cmp(b));
    let p = if n >= 500 {
        0.25
    } else {
        binom2(n as f64 / 2.0 + 1.0) / binom2(n as f64)
    };
    let q = quantile_sorted(&y, p).unwrap_or(f64::NAN);
    0.5 * (2.219 * q).powi(2)
}

fn shannon_entropy(x: &[f64], edges: &[f64]) -> f64 {
    let counts = histogram_counts(x, edges);
    let total: usize = counts.iter().sum();
    counts
        .iter()
        .map(|&c| c as f64 / total as f64 + 1e-15)
        .map(|p| -(p.log2() * p))
        .sum()
}

pub(crate) fn estimate(estimator: Estimator, x: &[f64], entropy_edges: &[f64]) -> f64 {
    let n = x.len() as f64;
    match estimator {
        Estimator::Matheron if x.is_empty() => f64::NAN,
        Estimator::Matheron => (1.0 / (2.0 * n)) * x.iter().map(|v| v * v).sum::<f64>(),
        Estimator::Cressie if x.is_empty() => f64::NAN,
        Estimator::Cressie => {
            let nominator = ((1.0 / n) * x.iter().map(|v| v.sqrt()).sum::<f64>()).powi(4);
            let denominator = 0.457 + (0.494 / n) + (0.045 / (n * n));
            nominator / (2.0 * denominator)
        }
        Estimator::Dowd => nanmedian(x).map_or(f64::NAN, |m| 2.198 * m * m / 2.0),
        Estimator::Genton => genton(x),
        Estimator::Minmax => {
            let v: Vec<f64> = x.iter().copied().filter(|v| !v.is_nan()).collect();
            if v.is_empty() {
                return f64::NAN;
            }
            let lo = v.iter().copied().fold(f64::INFINITY, f64::min);
            (max_of(&v) - lo) / mean(&v).unwrap_or(f64::NAN)
        }
        Estimator::Percentile(p) => nanpercentile(x, p).unwrap_or(f64::NAN),
        Estimator::Entropy(_) => shannon_entropy(x, entropy_edges),
    }
}

/// The pairs of a directional variogram, in `pdist` order, and the resolved bandwidth. The
/// angle of each pair is taken from the Euclidean coordinates whatever the metric, as in
/// scikit-gstat; coincident points have no direction and are left out.
fn direction_mask(
    coords: &[f64],
    dim: usize,
    direction: &Direction,
    distances: &[f64],
) -> Result<(Vec<bool>, f64), String> {
    let Direction {
        azimuth,
        tolerance,
        bandwidth,
        area,
    } = *direction;
    if !(-180.0..=180.0).contains(&azimuth) {
        return Err("the azimuth must lie in [-180, 180] degrees".to_string());
    }
    if !(0.0..=360.0).contains(&tolerance) {
        return Err("the tolerance must lie in [0, 360] degrees".to_string());
    }
    if dim > 2 {
        return Err("a directional variogram needs 1D or 2D coordinates".to_string());
    }
    let width = match bandwidth {
        Bandwidth::Value(w) if w >= 0.0 => w,
        Bandwidth::Value(_) => return Err("the bandwidth cannot be negative".to_string()),
        Bandwidth::Percentile(q) => {
            nanpercentile(distances, q).ok_or_else(|| "no pair distances".to_string())?
        }
    };
    let unbounded = width > max_of(distances);
    let xy = |i: usize| {
        let x = coords[i * dim];
        (x, if dim == 2 { coords[i * dim + 1] } else { 0.0 })
    };
    let n = coords.len() / dim;
    let shift = azimuth.to_radians();
    let half_tolerance = (tolerance / 2.0).to_radians();
    let mut mask = Vec::with_capacity(distances.len());
    for i in 0..n {
        let (xi, yi) = xy(i);
        for j in (i + 1)..n {
            let (xj, yj) = xy(j);
            let (dx, dy) = (xi - xj, yi - yj);
            let d = (dx * dx + dy * dy).sqrt();
            let pos = (dx / d).acos();
            let angle = if dy >= 0.0 { pos } else { -pos };
            let off = (angle + shift).abs();
            let mut diff = if off > std::f64::consts::PI {
                off - std::f64::consts::PI
            } else {
                off
            };
            if diff > std::f64::consts::FRAC_PI_2 {
                diff = std::f64::consts::PI - diff;
            }
            let in_tolerance = diff <= half_tolerance;
            let in_band = match area {
                SearchArea::Compass => true,
                SearchArea::Triangle => unbounded || width / 2.0 >= (d * off.sin()).abs(),
            };
            mask.push(in_tolerance && in_band);
        }
    }
    Ok((mask, width))
}

/// The experimental variogram of `values` at row-major coordinates of dimension `dim`.
pub fn experimental(
    coords: &[f64],
    dim: usize,
    values: &[f64],
    opts: &Options,
) -> Result<Experimental, String> {
    let Options {
        metric,
        maxlag,
        bins,
        n_lags,
        estimator,
        direction,
        cross,
        sampling,
        entropy_sigma,
    } = opts.clone();
    if dim == 0 || !coords.len().is_multiple_of(dim) || coords.len() / dim != values.len() {
        return Err("coordinates and values must describe the same points".to_string());
    }
    if values.len() < 2 {
        return Err("a variogram needs at least 2 points".to_string());
    }
    if n_lags == 0 {
        return Err("n_lags must be a positive integer".to_string());
    }
    match &cross {
        Some(Cross::Skgstat(co) | Cross::Classical(co)) if co.len() != values.len() => {
            return Err("the co-variable needs one value per point".to_string());
        }
        Some(Cross::Classical(_)) if estimator != Estimator::Matheron => {
            return Err("the classical cross-variogram needs the Matheron estimator".to_string());
        }
        _ => {}
    }
    // The value term of a pair: |a_i - a_j|, scikit-gstat's |a_i - a_j| |b_i - b_j| or the
    // classical (a_i - a_j) (b_i - b_j).
    let diff = |i: usize, j: usize| match &cross {
        None => (values[i] - values[j]).abs(),
        Some(Cross::Skgstat(co)) => (values[i] - values[j]).abs() * (co[i] - co[j]).abs(),
        Some(Cross::Classical(co)) => (values[i] - values[j]) * (co[i] - co[j]),
    };
    let (distances, diffs): (Vec<f64>, Vec<f64>) = match sampling {
        None => {
            let n = values.len();
            // Per pair: its distance and value term, a directional copy and mask, and the copies
            // the lags and maxlag take (about 30 bytes measured at 10k points).
            memory::ensure(40.0 * n as f64 * (n as f64 - 1.0) / 2.0, || {
                format!("a variogram of every pair of {n} points")
            })
            .map_err(|e| format!("{e}; use samples: to bound the pairs"))?;
            let diffs = (0..n)
                .flat_map(|i| ((i + 1)..n).map(move |j| (i, j)))
                .map(|(i, j)| diff(i, j))
                .collect();
            (pdist(coords, dim, metric)?, diffs)
        }
        Some(s) => {
            if metric != Metric::Euclidean {
                return Err("pair sampling needs the euclidean metric".to_string());
            }
            if direction.is_some() {
                return Err("pair sampling cannot be combined with a direction".to_string());
            }
            let pairs = sampled_pairs(coords, dim, s)?;
            let diffs = pairs.iter().map(|&(l, r, _)| diff(l, r)).collect();
            (pairs.into_iter().map(|(_, _, d)| d).collect(), diffs)
        }
    };
    let (mask, bandwidth) = match &direction {
        Some(dir) => {
            let (mask, width) = direction_mask(coords, dim, dir, &distances)?;
            (Some(mask), Some(width))
        }
        None => (None, None),
    };
    // The lags of a directional variogram come from its own pairs (the others are NaN); its
    // `maxlag` still from all of them.
    let lag_distances: Vec<f64> = match &mask {
        Some(m) => distances
            .iter()
            .zip(m)
            .map(|(d, keep)| if *keep { *d } else { f64::NAN })
            .collect(),
        None => distances.clone(),
    };
    let edges = lag_edges(
        &bins,
        &lag_distances,
        n_lags,
        resolve_maxlag(maxlag, &distances),
    )?;
    // Explicit edges set `maxlag` to the last edge, as scikit-gstat does.
    let maxlag = match bins {
        Bins::Edges(ref e) => e.last().copied(),
        _ => resolve_maxlag(maxlag, &distances),
    };
    if edges.is_empty() {
        return Err("there must be at least one lag".to_string());
    }
    let mut groups: Vec<Vec<f64>> = vec![Vec::new(); edges.len()];
    for (d, diff) in lag_distances.iter().zip(&diffs) {
        if d.is_nan() {
            continue;
        }
        if let Some(g) = group_of(*d, &edges) {
            groups[g].push(*diff);
        }
    }
    let entropy_edges = match estimator {
        Estimator::Entropy(n) => {
            let n = n.saturating_sub(1).max(1);
            let (first, last) = outer_edges(&distances);
            linspace(first, last, n + 1)
        }
        _ => Vec::new(),
    };
    let classical = matches!(cross, Some(Cross::Classical(_)));
    let entropy_sigma = if entropy_sigma {
        let edges = rule_edges(Rule::Scott, &distances)?;
        Some(
            groups
                .iter()
                .map(|g| {
                    if g.is_empty() {
                        f64::NAN
                    } else {
                        1.0 / shannon_entropy(g, &edges)
                    }
                })
                .collect(),
        )
    } else {
        None
    };
    Ok(Experimental {
        experimental: groups
            .iter()
            .map(|g| match g.len() {
                0 if classical => f64::NAN,
                n if classical => g.iter().sum::<f64>() / (2.0 * n as f64),
                _ => estimate(estimator, g, &entropy_edges),
            })
            .collect(),
        bin_count: groups.iter().map(Vec::len).collect(),
        bins: edges,
        maxlag,
        bandwidth,
        entropy_sigma,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts(bins: Bins, n_lags: usize, estimator: Estimator) -> Options {
        Options {
            metric: Metric::Euclidean,
            maxlag: MaxLag::Unset,
            bins,
            n_lags,
            estimator,
            direction: None,
            cross: None,
            sampling: None,
            entropy_sigma: false,
        }
    }

    fn line() -> (Vec<f64>, Vec<f64>) {
        // Four points on a line at 0, 1, 2, 3 with values 1, 2, 4, 7.
        (
            vec![0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0],
            vec![1.0, 2.0, 4.0, 7.0],
        )
    }

    #[test]
    fn pdist_order_and_metrics() {
        let c = [0.0, 0.0, 3.0, 4.0, 1.0, 1.0];
        assert_eq!(pdist(&c, 2, Metric::Euclidean).unwrap()[0], 5.0);
        assert_eq!(
            pdist(&c, 2, Metric::Cityblock).unwrap(),
            vec![7.0, 2.0, 5.0]
        );
        assert_eq!(
            pdist(&c, 2, Metric::Chebyshev).unwrap(),
            vec![4.0, 1.0, 3.0]
        );
        assert_eq!(pairwise_diffs(&[1.0, 4.0, 2.0]), vec![3.0, 1.0, 2.0]);
    }

    #[test]
    fn matheron_by_hand_and_the_pair_at_maxlag_is_left_out() {
        let (c, z) = line();
        // Distances 1,2,3,1,2,1; even edges at 1.5 and 3 (maxlag = largest distance).
        let v = experimental(&c, 2, &z, &opts(Bins::Even, 2, Estimator::Matheron)).unwrap();
        assert_eq!(v.bins, vec![1.5, 3.0]);
        // Lag 0: d = 1 pairs (1,2), (2,4), (4,7) -> diffs 1, 2, 3; lag 1: d = 2 pairs, diffs 3, 5.
        assert_eq!(v.bin_count, vec![3, 2]);
        assert!((v.experimental[0] - 14.0 / 6.0).abs() < 1e-15);
        assert!((v.experimental[1] - 34.0 / 4.0).abs() < 1e-15);
    }

    #[test]
    fn entropy_sigma_by_hand() {
        let (c, z) = line();
        let mut o = opts(Bins::Even, 2, Estimator::Matheron);
        o.entropy_sigma = true;
        let v = experimental(&c, 2, &z, &o).unwrap();
        // Scott's rule over the distances 1, 2, 3, 1, 2, 1 gives the edges 1, 2, 3; lag 0 holds
        // the differences 1, 2, 3, counted 1 and 2.
        let h = -(1.0_f64 / 3.0 * (1.0_f64 / 3.0).log2() + 2.0 / 3.0 * (2.0_f64 / 3.0).log2());
        let sigma = v.entropy_sigma.unwrap();
        assert_eq!(sigma.len(), 2);
        assert!((sigma[0] - 1.0 / h).abs() < 1e-12, "{}", sigma[0]);
        assert!(
            experimental(&c, 2, &z, &opts(Bins::Even, 2, Estimator::Matheron))
                .unwrap()
                .entropy_sigma
                .is_none()
        );
    }

    #[test]
    fn sampling_every_point_keeps_every_pair() {
        // With k = n both subsets hold all points, so the pairs r < l are all the pairs.
        let (c, z) = line();
        let dense = experimental(&c, 2, &z, &opts(Bins::Even, 2, Estimator::Matheron)).unwrap();
        let mut o = opts(Bins::Even, 2, Estimator::Matheron);
        o.sampling = Some(Sampling { k: 4, seed: 42 });
        let sampled = experimental(&c, 2, &z, &o).unwrap();
        assert_eq!(sampled.bin_count, dense.bin_count);
        assert_eq!(sampled.bins, dense.bins);
        for (a, b) in sampled.experimental.iter().zip(&dense.experimental) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn sampling_refuses_what_scikit_gstat_cannot_do() {
        let (c, z) = line();
        let sampled = |k: usize, metric: Metric, direction: Option<Direction>| {
            let mut o = opts(Bins::Even, 2, Estimator::Matheron);
            o.sampling = Some(Sampling { k, seed: 1 });
            o.metric = metric;
            o.direction = direction;
            experimental(&c, 2, &z, &o)
        };
        assert!(sampled(0, Metric::Euclidean, None).is_err());
        assert!(sampled(5, Metric::Euclidean, None).is_err());
        assert!(sampled(3, Metric::Cityblock, None).is_err());
        let dir = Direction {
            azimuth: 0.0,
            tolerance: 45.0,
            bandwidth: Bandwidth::Value(1.0),
            area: SearchArea::Triangle,
        };
        assert!(sampled(3, Metric::Euclidean, Some(dir)).is_err());
    }

    #[test]
    fn cross_variograms_by_hand() {
        let (c, a) = line();
        let b = vec![2.0, 1.0, 5.0, 3.0];
        // Lag 0 holds the pairs (0,1), (1,2), (2,3): da = -1, -2, -3 and db = 1, -4, 2; lag 1
        // holds (0,2), (1,3): da = -3, -5 and db = -3, -2.
        let mut o = opts(Bins::Even, 2, Estimator::Matheron);
        o.cross = Some(Cross::Skgstat(b.clone()));
        let v = experimental(&c, 2, &a, &o).unwrap();
        assert_eq!(v.bin_count, vec![3, 2]);
        // |da| |db| = 1, 8, 6 and 9, 10, squared by Matheron.
        assert!((v.experimental[0] - 101.0 / 6.0).abs() < 1e-12);
        assert!((v.experimental[1] - 181.0 / 4.0).abs() < 1e-12);

        o.cross = Some(Cross::Classical(b.clone()));
        let v = experimental(&c, 2, &a, &o).unwrap();
        // sum(da db) / 2N: (-1 + 8 - 6) / 6 and (9 + 10) / 4.
        assert!((v.experimental[0] - 1.0 / 6.0).abs() < 1e-15);
        assert!((v.experimental[1] - 19.0 / 4.0).abs() < 1e-15);

        let mut cressie = opts(Bins::Even, 2, Estimator::Cressie);
        cressie.cross = Some(Cross::Classical(b));
        assert!(experimental(&c, 2, &a, &cressie).is_err());
        o.cross = Some(Cross::Skgstat(vec![1.0, 2.0]));
        assert!(experimental(&c, 2, &a, &o).is_err());
    }

    #[test]
    fn scipy_metrics_by_hand() {
        let close = |a: &[f64], b: &[f64]| a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-14);
        // (1, 2) and (4, 6): differences 3 and 4.
        let ab = [1.0, 2.0, 4.0, 6.0];
        assert_eq!(pdist(&ab, 2, Metric::SqEuclidean).unwrap(), vec![25.0]);
        assert_eq!(pdist(&ab, 2, Metric::Minkowski).unwrap(), vec![5.0]);
        assert!(close(
            &pdist(&ab, 2, Metric::Canberra).unwrap(),
            &[3.0 / 5.0 + 4.0 / 8.0]
        ));
        assert!(close(
            &pdist(&ab, 2, Metric::Braycurtis).unwrap(),
            &[7.0 / 13.0]
        ));
        assert!(close(
            &pdist(&ab, 2, Metric::Cosine).unwrap(),
            &[1.0 - 16.0 / (5.0_f64 * 52.0).sqrt()]
        ));
        // Canberra skips a coordinate where both are 0, as scipy does.
        assert_eq!(
            pdist(&[0.0, 2.0, 0.0, -2.0], 2, Metric::Canberra).unwrap(),
            vec![1.0]
        );
        // (0, 0), (2, 0), (0, 4): column variances (ddof 1) 4/3 and 16/3, and the inverse
        // covariance [[1, 1/4], [1/4, 1/4]] makes every pair 2 apart.
        let tri = [0.0, 0.0, 2.0, 0.0, 0.0, 4.0];
        let s3 = 3.0_f64.sqrt();
        let (dx, dy): (f64, f64) = (2.0 * 2.0 / (4.0 / 3.0), 4.0 * 4.0 / (16.0 / 3.0));
        assert!(close(
            &pdist(&tri, 2, Metric::SEuclidean).unwrap(),
            &[s3, s3, (dx + dy).sqrt()]
        ));
        assert!(close(
            &pdist(&tri, 2, Metric::Mahalanobis).unwrap(),
            &[2.0, 2.0, 2.0]
        ));
    }

    #[test]
    fn maxlag_forms() {
        let d = [1.0, 2.0, 3.0, 10.0];
        assert_eq!(resolve_maxlag(MaxLag::Value(0.5), &d), Some(5.0));
        assert_eq!(resolve_maxlag(MaxLag::Value(4.0), &d), Some(4.0));
        assert_eq!(resolve_maxlag(MaxLag::Mean, &d), Some(4.0));
        assert_eq!(resolve_maxlag(MaxLag::Median, &d), Some(2.5));
        assert_eq!(resolve_maxlag(MaxLag::Unset, &d), None);
    }

    #[test]
    fn scipy_binom_of_a_real_argument() {
        assert_eq!(binom2(5.0), 10.0);
        assert_eq!(binom2(2.0), 1.0);
        assert!((binom2(3.5) - 3.5 * 2.5 / 2.0).abs() < 1e-15);
    }

    #[test]
    fn empty_lags_are_nan_and_bad_input_is_an_error() {
        let (c, z) = line();
        let edges = Bins::Edges(vec![0.5, 0.9, 3.5]);
        // Explicit edges set the number of lags; `n_lags` is ignored.
        let v = experimental(&c, 2, &z, &opts(edges, 1, Estimator::Cressie)).unwrap();
        assert!(v.experimental[0].is_nan() && v.experimental[1].is_nan());
        assert_eq!(v.bin_count, vec![0, 0, 6]);
        assert_eq!(v.maxlag, Some(3.5));
        assert!(experimental(
            &c[..2],
            2,
            &z[..1],
            &opts(Bins::Even, 2, Estimator::Matheron)
        )
        .is_err());
    }

    #[test]
    fn directions_select_pairs_along_the_azimuth() {
        // A cross: two pairs along x, two along y and the diagonals between them.
        let coords = [0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 4.0, 0.0];
        let distances = pdist(&coords, 2, Metric::Euclidean).unwrap();
        let dir = |azimuth, area| Direction {
            azimuth,
            tolerance: 10.0,
            bandwidth: Bandwidth::Value(1.0),
            area,
        };
        let (east, _) =
            direction_mask(&coords, 2, &dir(0.0, SearchArea::Compass), &distances).unwrap();
        // Pairs (0,1), (0,3) and (1,3) lie along x.
        assert_eq!(east, vec![true, false, true, false, true, false]);
        let (north, _) =
            direction_mask(&coords, 2, &dir(90.0, SearchArea::Triangle), &distances).unwrap();
        assert_eq!(north, vec![false, true, false, false, false, false]);
        // A bandwidth beyond every distance has no effect: the triangle is the compass.
        let wide = Direction {
            bandwidth: Bandwidth::Value(100.0),
            tolerance: 120.0,
            ..dir(0.0, SearchArea::Triangle)
        };
        let compass = Direction {
            area: SearchArea::Compass,
            ..wide
        };
        assert_eq!(
            direction_mask(&coords, 2, &wide, &distances).unwrap().0,
            direction_mask(&coords, 2, &compass, &distances).unwrap().0
        );
        assert!(direction_mask(&coords, 2, &dir(200.0, SearchArea::Compass), &distances).is_err());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn a_pair_list_too_large_returns_an_error_before_allocating() {
        let n = 300_000;
        let coords: Vec<f64> = (0..2 * n).map(|i| (i * 7 % 1000) as f64).collect();
        let values = vec![1.0; n];
        let err = experimental(
            &coords,
            2,
            &values,
            &opts(Bins::Even, 10, Estimator::Matheron),
        )
        .unwrap_err();
        assert!(
            err.contains("variogram of every pair of 300000 points") && err.contains("samples"),
            "{err}"
        );
        // Sampled pairs stay small.
        let sampled = Options {
            sampling: Some(Sampling { k: 200, seed: 1 }),
            ..opts(Bins::Even, 10, Estimator::Matheron)
        };
        assert!(experimental(&coords, 2, &values, &sampled).is_ok());
    }
}
