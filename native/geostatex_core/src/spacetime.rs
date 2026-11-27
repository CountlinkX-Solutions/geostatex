//! Space-time experimental variograms after `skgstat.SpaceTimeVariogram`: pairs of observations
//! at two points and two time steps, classed by their spatial and temporal lags.

use crate::memory;
use crate::variogram::{
    estimate, lag_edges, pdist, resolve_maxlag, Bins, Estimator, MaxLag, Metric,
};

/// How a space-time variogram is computed (`skgstat.SpaceTimeVariogram` arguments).
#[derive(Clone, Debug, PartialEq)]
pub struct Options {
    pub metric: Metric,
    pub x_bins: Bins,
    pub x_lags: usize,
    pub maxlag: MaxLag,
    pub t_bins: Bins,
    /// `None` for one lag per time step (`t_lags="max"`).
    pub t_lags: Option<usize>,
    pub estimator: Estimator,
}

/// A space-time experimental variogram.
#[derive(Clone, Debug, PartialEq)]
pub struct SpaceTime {
    /// Upper edges of the spatial lags.
    pub xbins: Vec<f64>,
    /// Upper edges of the temporal lags.
    pub tbins: Vec<f64>,
    /// Semivariance by spatial lag (outer) and temporal lag (inner); NaN for an empty class.
    pub experimental: Vec<f64>,
    pub counts: Vec<usize>,
    /// The resolved spatial `maxlag`.
    pub maxlag: Option<f64>,
}

/// scikit-gstat's space-time lag classes: `(lower, upper]` with the first lower edge 0, so a
/// zero distance belongs to no class.
fn classes(distances: &[f64], edges: &[f64]) -> Vec<Option<usize>> {
    distances
        .iter()
        .map(|&d| {
            let i = edges.partition_point(|&e| e < d);
            (d > 0.0 && i < edges.len()).then_some(i)
        })
        .collect()
}

/// The space-time variogram of `values` (`n` points by `n_times` steps, row-major) at
/// row-major coordinates of dimension `dim`. A pair takes the first point at the earlier step
/// and the second point at the later one (`i < j`, `ti < tj`), as scikit-gstat does.
pub fn experimental(
    coords: &[f64],
    dim: usize,
    values: &[f64],
    n_times: usize,
    opts: &Options,
) -> Result<SpaceTime, String> {
    if dim == 0 || !coords.len().is_multiple_of(dim) {
        return Err("coordinates must have the given dimension".to_string());
    }
    let n = coords.len() / dim;
    if n < 2 || n_times < 2 || values.len() != n * n_times {
        return Err("values must have one row of at least 2 time steps per point".to_string());
    }
    if matches!(opts.estimator, Estimator::Entropy(_)) {
        return Err("the entropy estimator is not available for space-time variograms".to_string());
    }
    if opts.x_lags == 0 || opts.t_lags == Some(0) {
        return Err("x_lags and t_lags must be positive".to_string());
    }
    // The point pairs' distances, indices and classes, and one class's value pairs at a time
    // (at most every point pair with every time pair).
    let point_pairs = n as f64 * (n as f64 - 1.0) / 2.0;
    let time_pairs = n_times as f64 * (n_times as f64 - 1.0) / 2.0;
    memory::ensure(point_pairs * (48.0 + 8.0 * time_pairs), || {
        format!("a space-time variogram of {n} points and {n_times} time steps")
    })?;
    let xdist = pdist(coords, dim, opts.metric)?;
    let maxlag = resolve_maxlag(opts.maxlag, &xdist);
    let xbins = lag_edges(&opts.x_bins, &xdist, opts.x_lags, maxlag)?;
    let times: Vec<f64> = (0..n_times).flat_map(|t| [t as f64, 0.0]).collect();
    let tdist = pdist(&times, 2, opts.metric)?;
    let tbins = lag_edges(
        &opts.t_bins,
        &tdist,
        opts.t_lags.unwrap_or(n_times - 1),
        None,
    )?;
    if xbins.is_empty() || tbins.is_empty() {
        return Err("there must be at least one lag on each axis".to_string());
    }
    let xclass = classes(&xdist, &xbins);
    let tclass = classes(&tdist, &tbins);
    let pairs = |m: usize| -> Vec<(usize, usize)> {
        (0..m)
            .flat_map(|i| ((i + 1)..m).map(move |j| (i, j)))
            .collect()
    };
    let (xpairs, tpairs) = (pairs(n), pairs(n_times));
    let rows_of = |k: usize| -> Vec<usize> {
        (0..xpairs.len())
            .filter(|&r| xclass[r] == Some(k))
            .collect()
    };
    let cols_of = |k: usize| -> Vec<usize> {
        (0..tpairs.len())
            .filter(|&c| tclass[c] == Some(k))
            .collect()
    };
    let cols: Vec<Vec<usize>> = (0..tbins.len()).map(cols_of).collect();
    let mut experimental = Vec::with_capacity(xbins.len() * tbins.len());
    let mut counts = Vec::with_capacity(experimental.capacity());
    for x in 0..xbins.len() {
        let rows = rows_of(x);
        for tc in &cols {
            let mut class = Vec::with_capacity(rows.len() * tc.len());
            for &r in &rows {
                let (xi, xj) = xpairs[r];
                for &c in tc {
                    let (ti, tj) = tpairs[c];
                    class.push((values[xi * n_times + ti] - values[xj * n_times + tj]).abs());
                }
            }
            counts.push(class.len());
            experimental.push(estimate(opts.estimator, &class, &[]));
        }
    }
    Ok(SpaceTime {
        xbins,
        tbins,
        experimental,
        counts,
        maxlag,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classes_exclude_zero_and_include_the_upper_edge() {
        assert_eq!(
            classes(&[0.0, 1.0, 1.5, 2.0, 2.5], &[1.0, 2.0]),
            vec![None, Some(0), Some(1), Some(1), None]
        );
    }

    #[test]
    fn pairs_take_the_earlier_step_at_the_first_point() {
        // Two points 1 apart, three steps; values grow by 1 per step at point 0, by 10 at point 1.
        let coords = [0.0, 0.0, 1.0, 0.0];
        let values = [0.0, 1.0, 2.0, 0.0, 10.0, 20.0];
        let opts = Options {
            metric: Metric::Euclidean,
            x_bins: Bins::Even,
            x_lags: 1,
            maxlag: MaxLag::Unset,
            t_bins: Bins::Even,
            t_lags: None,
            estimator: Estimator::Matheron,
        };
        let st = experimental(&coords, 2, &values, 3, &opts).unwrap();
        assert_eq!(st.tbins, vec![1.0, 2.0]);
        // Lag 1: |v0[0] - v1[1]| = 10, |v0[1] - v1[2]| = 19; lag 2: |v0[0] - v1[2]| = 20.
        assert_eq!(st.counts, vec![2, 1]);
        assert_eq!(st.experimental, vec![(100.0 + 361.0) / 4.0, 400.0 / 2.0]);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn a_pair_list_too_large_returns_an_error_before_allocating() {
        let n = 300_000;
        let coords: Vec<f64> = (0..2 * n).map(|i| (i * 7 % 1000) as f64).collect();
        let values = vec![1.0; 2 * n];
        let opts = Options {
            metric: Metric::Euclidean,
            x_bins: Bins::Even,
            x_lags: 10,
            maxlag: MaxLag::Unset,
            t_bins: Bins::Even,
            t_lags: None,
            estimator: Estimator::Matheron,
        };
        let err = experimental(&coords, 2, &values, 2, &opts).unwrap_err();
        assert!(
            err.contains("space-time variogram of 300000 points and 2 time steps needs about"),
            "{err}"
        );
    }
}
