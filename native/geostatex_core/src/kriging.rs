//! Ordinary and universal kriging after `pykrige.OrdinaryKriging` / `UniversalKriging`: PyKrige's variogram models and automatic
//! variogram, the kriging system with a Lagrange multiplier, exact values, pseudo-inverses, a
//! moving window, anisotropy and great-circle distances.

use crate::kdtree::KdTree;
use crate::lsq::{minimise, solve_small};
use crate::memory;
use crate::parallel::{par_map, par_map_dynamic};
use faer::linalg::solvers::{DenseSolveCore, Solve};
use faer::{Mat, Side};

/// PyKrige's variogram models.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Model {
    /// `[slope, nugget]`.
    Linear,
    /// `[scale, exponent, nugget]`.
    Power,
    /// `[psill, range, nugget]` for the four below.
    Gaussian,
    Spherical,
    Exponential,
    HoleEffect,
    /// A GSTools model: `[psill, range, nugget, shapes...]`, `range` its length scale
    /// (`rescale = 1`).
    Gstools(crate::gstools::GsModel),
    /// PyKrige's `"custom"`: a user function, given to the model as a table of its values at
    /// the distances kriging needs (`Kriging::gamma_table`); `params` are the user's.
    Custom,
}

/// Semivariance of `model` with PyKrige's parameters at distance `d`.
pub fn value(model: Model, m: &[f64], d: f64) -> f64 {
    match model {
        Model::Linear => m[0] * d + m[1],
        Model::Power => m[0] * d.powf(m[1]) + m[2],
        Model::Gaussian => {
            m[0] * (1.0 - (-d.powf(2.0) / (m[1] * 4.0 / 7.0).powf(2.0)).exp()) + m[2]
        }
        Model::Exponential => m[0] * (1.0 - (-d / (m[1] / 3.0)).exp()) + m[2],
        Model::Spherical => {
            if d <= m[1] {
                m[0] * (3.0 * d / (2.0 * m[1]) - d.powf(3.0) / (2.0 * m[1].powf(3.0))) + m[2]
            } else {
                m[0] + m[2]
            }
        }
        Model::HoleEffect => {
            m[0] * (1.0 - (1.0 - d / (m[1] / 3.0)) * (-d / (m[1] / 3.0)).exp()) + m[2]
        }
        Model::Gstools(g) => m[0] * (1.0 - crate::gstools::cor(g, d / m[1], &m[3..])) + m[2],
        // Only through a gamma table.
        Model::Custom => f64::NAN,
    }
}

/// PyKrige's `great_circle_distance` in degrees.
pub fn great_circle(lon1: f64, lat1: f64, lon2: f64, lat2: f64) -> f64 {
    let (lat1, lat2) = (
        lat1 * std::f64::consts::PI / 180.0,
        lat2 * std::f64::consts::PI / 180.0,
    );
    let dlon = (lon1 - lon2) * std::f64::consts::PI / 180.0;
    let (c1, s1, c2, s2, cd) = (lat1.cos(), lat1.sin(), lat2.cos(), lat2.sin(), dlon.cos());
    180.0 / std::f64::consts::PI
        * ((c2 * dlon.sin()).powi(2) + (c1 * s2 - s1 * c2 * cd).powi(2))
            .sqrt()
            .atan2(s1 * s2 + c1 * c2 * cd)
}

/// A point in 2D (`z = 0`) or 3D.
pub type Point = [f64; 3];

fn matmul(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    out
}

/// PyKrige's `_adjust_for_anisotropy` in 3D (2D is the case `z = 0`, angles `[0, 0, angle]`
/// and scalings `[scaling, 1]`): rotate by `-angle_x`, `-angle_y`, `-angle_z` degrees about
/// `center` (`Rz Ry Rx`), then stretch y and z by `scaling`.
pub fn adjust(points: &[Point], center: Point, scaling: [f64; 2], angles: [f64; 3]) -> Vec<Point> {
    let a = angles.map(|v| -v * std::f64::consts::PI / 180.0);
    let rx = [
        [1.0, 0.0, 0.0],
        [0.0, a[0].cos(), -a[0].sin()],
        [0.0, a[0].sin(), a[0].cos()],
    ];
    let ry = [
        [a[1].cos(), 0.0, a[1].sin()],
        [0.0, 1.0, 0.0],
        [-a[1].sin(), 0.0, a[1].cos()],
    ];
    let rz = [
        [a[2].cos(), -a[2].sin(), 0.0],
        [a[2].sin(), a[2].cos(), 0.0],
        [0.0, 0.0, 1.0],
    ];
    let rot = matmul(&rz, &matmul(&ry, &rx));
    let stretch = [1.0, scaling[0], scaling[1]];
    points
        .iter()
        .map(|p| {
            let v = [p[0] - center[0], p[1] - center[1], p[2] - center[2]];
            let r: Vec<f64> = (0..3)
                .map(|i| rot[i][0] * v[0] + rot[i][1] * v[1] + rot[i][2] * v[2])
                .collect();
            [
                stretch[0] * r[0] + center[0],
                stretch[1] * r[1] + center[1],
                stretch[2] * r[2] + center[2],
            ]
        })
        .collect()
}

/// How the kriging system is solved.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Inverse {
    /// LU inverse (`scipy.linalg.inv`).
    Exact,
    /// Moore-Penrose pseudo-inverse from the SVD (`scipy.linalg.pinv`).
    Pinv,
    /// Pseudo-inverse from the symmetric eigendecomposition (`scipy.linalg.pinvh`).
    Pinvh,
}

/// A drift term of universal kriging (`pykrige.UniversalKriging`).
#[derive(Clone, Debug, PartialEq)]
pub enum Drift {
    /// The (adjusted) x and y coordinates: two columns.
    RegionalLinear,
    /// The (adjusted) x, y and z coordinates: three columns (3D).
    RegionalLinear3,
    /// `-Q ln(distance)` to each well `[x, y, Q]` (adjusted), `-100` for a zero distance.
    PointLog(Vec<[f64; 3]>),
    /// Bilinear interpolation of a grid at the original coordinates: `x` (columns), `y` (rows),
    /// `z` row-major `y.len() x x.len()`.
    External {
        x: Vec<f64>,
        y: Vec<f64>,
        z: Vec<f64>,
    },
}

/// A kriging model over data points (anisotropy already applied): ordinary kriging without
/// drifts, universal kriging with them.
#[derive(Clone, Debug)]
pub struct Kriging {
    pub points: Vec<Point>,
    pub values: Vec<f64>,
    pub model: Model,
    pub params: Vec<f64>,
    pub geographic: bool,
    pub exact_values: bool,
    pub inverse: Inverse,
    pub center: Point,
    /// Anisotropy: scalings of y and z, rotation angles about x, y and z (degrees).
    pub scaling: [f64; 2],
    pub angles: [f64; 3],
    /// Drift terms evaluated by the model, in PyKrige's order.
    pub drifts: Vec<Drift>,
    /// Drift columns at the data points: the terms above, then the given (specified and
    /// functional) ones.
    pub data_drift: Vec<Vec<f64>>,
    /// The semivariances of a custom variogram keyed by the bits of each distance from
    /// [`Kriging::distances`] (the same computation), which replace the model when set.
    pub gamma_table: Option<std::collections::HashMap<u64, f64>>,
    /// The known mean of simple kriging (`gstools.krige.Simple`), `None` for ordinary and
    /// universal kriging. Simple kriging needs a model with a sill and no drift terms.
    pub simple_mean: Option<f64>,
}

fn point_log(p: Point, well: [f64; 3]) -> f64 {
    let l = ((p[0] - well[0]).powi(2) + (p[1] - well[1]).powi(2))
        .sqrt()
        .ln();
    -well[2] * if l.is_infinite() { -100.0 } else { l }
}

/// PyKrige's `_calculate_data_point_zscalars` at one point.
pub fn external_value(gx: &[f64], gy: &[f64], gz: &[f64], xn: f64, yn: f64) -> Result<f64, String> {
    let range = |v: &[f64]| {
        (
            v.iter().copied().fold(f64::INFINITY, f64::min),
            v.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        )
    };
    let ((minx, maxx), (miny, maxy)) = (range(gx), range(gy));
    if xn > maxx || xn < minx || yn > maxy || yn < miny {
        return Err("the external drift grid does not cover the point".to_string());
    }
    let first = |v: &[f64], t: f64| (0..v.len()).filter(|&i| v[i] >= t).min().unwrap_or(0);
    let last = |v: &[f64], t: f64| (0..v.len()).filter(|&i| v[i] <= t).max().unwrap_or(0);
    let (x1, x2, y1, y2) = (last(gx, xn), first(gx, xn), last(gy, yn), first(gy, yn));
    let nx = gx.len();
    let z = |r: usize, c: usize| gz[r * nx + c];
    Ok(if y1 == y2 {
        if x1 == x2 {
            z(y1, x1)
        } else {
            (z(y1, x1) * (gx[x2] - xn) + z(y2, x2) * (xn - gx[x1])) / (gx[x2] - gx[x1])
        }
    } else if x1 == x2 {
        (z(y1, x1) * (gy[y2] - yn) + z(y2, x2) * (yn - gy[y1])) / (gy[y2] - gy[y1])
    } else {
        (z(y1, x1) * (gx[x2] - xn) * (gy[y2] - yn)
            + z(y1, x2) * (xn - gx[x1]) * (gy[y2] - yn)
            + z(y2, x1) * (gx[x2] - xn) * (yn - gy[y1])
            + z(y2, x2) * (xn - gx[x1]) * (yn - gy[y1]))
            / ((gx[x2] - gx[x1]) * (gy[y2] - gy[y1]))
    })
}

/// The model's drift columns at points (`adjusted` for the regional and point-log terms,
/// `original` for the external grid), one per column.
pub fn drift_columns(
    drifts: &[Drift],
    adjusted: &[Point],
    original: &[Point],
) -> Result<Vec<Vec<f64>>, String> {
    let mut cols = Vec::new();
    for d in drifts {
        match d {
            Drift::RegionalLinear => {
                cols.push(adjusted.iter().map(|p| p[0]).collect());
                cols.push(adjusted.iter().map(|p| p[1]).collect());
            }
            Drift::RegionalLinear3 => {
                for k in 0..3 {
                    cols.push(adjusted.iter().map(|p| p[k]).collect());
                }
            }
            Drift::PointLog(wells) => {
                for w in wells {
                    cols.push(adjusted.iter().map(|p| point_log(*p, *w)).collect());
                }
            }
            Drift::External { x, y, z } => {
                let col: Result<Vec<f64>, String> = original
                    .iter()
                    .map(|p| external_value(x, y, z, p[0], p[1]))
                    .collect();
                cols.push(col?);
            }
        }
    }
    Ok(cols)
}

/// Targets per thread below which a prediction runs on fewer threads.
const MIN_TARGETS_PER_THREAD: usize = 16;

/// The unit vector of a longitude / latitude in degrees (PyKrige's KD-tree on the sphere).
fn unit_vector(q: Point) -> Point {
    let (lon, lat) = (q[0].to_radians(), q[1].to_radians());
    [lon.cos() * lat.cos(), lon.sin() * lat.cos(), lat.sin()]
}

/// `a_inv b`, read column by column (the matrix is column-major): each entry still adds its
/// terms in column order from the same start value as a sum, so it is the row-by-row sum to the
/// bit, without striding through the matrix.
fn times(a_inv: &Mat<f64>, b: &[f64]) -> Vec<f64> {
    let mut x = vec![std::iter::empty::<f64>().sum::<f64>(); b.len()];
    for (c, &bc) in b.iter().enumerate() {
        let col = a_inv
            .col(c)
            .try_as_col_major()
            .expect("an owned matrix is column-major")
            .as_slice();
        for (xr, &a) in x.iter_mut().zip(col) {
            *xr += a * bc;
        }
    }
    x
}

fn dist(a: Point, b: Point, geographic: bool) -> f64 {
    if geographic {
        great_circle(a[0], a[1], b[0], b[1])
    } else {
        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
    }
}

/// PyKrige's experimental variogram: `nlags` equal-width lags from the smallest to the largest
/// distance (the last one stretched by 0.001), the mean distance and the mean semivariance of
/// each non-empty lag.
///
/// The pairs are not stored: a first pass finds the distance range, a second one adds each pair
/// to the one lag `[bins[k], bins[k + 1])` holding it. Each lag accumulates its pairs in the same
/// order as PyKrige's per-lag selection, so the sums are the same to the bit.
pub fn lags(
    points: &[Point],
    values: &[f64],
    nlags: usize,
    geographic: bool,
) -> (Vec<f64>, Vec<f64>) {
    let n = points.len();
    let (mut dmin, mut dmax) = (f64::INFINITY, f64::NEG_INFINITY);
    for i in 0..n {
        for j in (i + 1)..n {
            let d = dist(points[i], points[j], geographic);
            dmax = dmax.max(d);
            dmin = dmin.min(d);
        }
    }
    let dd = (dmax - dmin) / nlags as f64;
    let mut bins: Vec<f64> = (0..nlags).map(|k| dmin + k as f64 * dd).collect();
    bins.push(dmax + 0.001);
    let (mut sd, mut sg, mut cnt) = (vec![0.0; nlags], vec![0.0; nlags], vec![0usize; nlags]);
    for i in 0..n {
        for j in (i + 1)..n {
            let d = dist(points[i], points[j], geographic);
            // The edges never decrease, so the last edge <= d names the only lag that can hold
            // d; a NaN distance matches none.
            let k = bins.partition_point(|&e| e <= d);
            if k == 0 || k > nlags || d >= bins[k] {
                continue;
            }
            sd[k - 1] += d;
            sg[k - 1] += 0.5 * (values[i] - values[j]).powi(2);
            cnt[k - 1] += 1;
        }
    }
    let (mut out_lags, mut out_sv) = (Vec::new(), Vec::new());
    for k in 0..nlags {
        if cnt[k] > 0 {
            out_lags.push(sd[k] / cnt[k] as f64);
            out_sv.push(sg[k] / cnt[k] as f64);
        }
    }
    (out_lags, out_sv)
}

/// PyKrige's sigmoid lag weights (`weight=True`), normalised to sum 1.
fn lag_weights(lags: &[f64], weight: bool) -> Vec<f64> {
    if !weight {
        return vec![1.0; lags.len()];
    }
    let max = lags.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let min = lags.iter().copied().fold(f64::INFINITY, f64::min);
    let drange = max - min;
    let k = 2.1972 / (0.1 * drange);
    let x0 = 0.7 * drange + min;
    let w: Vec<f64> = lags
        .iter()
        .map(|x| 1.0 / (1.0 + (-k * (x0 - x)).exp()))
        .collect();
    let total: f64 = w.iter().sum();
    w.iter().map(|v| v / total).collect()
}

/// The `soft_l1` objective PyKrige's fit minimises: `sum(2 (sqrt(1 + r^2) - 1))`.
pub fn objective(model: Model, params: &[f64], lags: &[f64], sv: &[f64], weight: bool) -> f64 {
    let w = lag_weights(lags, weight);
    lags.iter()
        .zip(sv)
        .zip(&w)
        .map(|((&l, &s), &wt)| {
            let r = (value(model, params, l) - s) * wt;
            2.0 * ((1.0 + r * r).sqrt() - 1.0)
        })
        .sum()
}

/// PyKrige's parameter bounds for the automatic fit.
fn bounds(model: Model, lags: &[f64], sv: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let max_sv = sv.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let max_lag = lags.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    match model {
        Model::Linear => (vec![0.0, 0.0], vec![f64::INFINITY, max_sv]),
        Model::Power => (vec![0.0, 0.001, 0.0], vec![f64::INFINITY, 1.999, max_sv]),
        _ => (vec![0.0, 0.0, 0.0], vec![10.0 * max_sv, max_lag, max_sv]),
    }
}

/// Fits PyKrige's automatic variogram: a scan of the non-linear parameter (range or exponent)
/// with the linear ones solved by robust (soft_l1) reweighted least squares, refined by a bounded
/// Levenberg-Marquardt on the soft_l1 objective.
pub fn fit(model: Model, lags: &[f64], sv: &[f64], weight: bool) -> Result<Vec<f64>, String> {
    if let Model::Gstools(_) = model {
        return Err(
            "GSTools models need :variogram_parameters (PyKrige has no automatic fit for them)"
                .to_string(),
        );
    }
    if model == Model::Custom {
        return Err("a custom variogram needs its variogram_parameters".to_string());
    }
    if lags.len() < 2 {
        return Err("the automatic variogram needs at least 2 non-empty lags".to_string());
    }
    let (lb, ub) = bounds(model, lags, sv);
    let w = lag_weights(lags, weight);
    let transformed = |p: &[f64]| -> Vec<f64> {
        lags.iter()
            .zip(sv)
            .zip(&w)
            .map(|((&l, &s), &wt)| {
                let r = (value(model, p, l) - s) * wt;
                let rho = 2.0 * ((1.0 + r * r).sqrt() - 1.0);
                rho.sqrt() * r.signum()
            })
            .collect()
    };
    let max_sv = sv.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let max_lag = lags.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    // Candidate values of the non-linear parameter; the linear ones (a multiplier and a nugget)
    // come from robust reweighted least squares.
    let nonlinear: Vec<Option<f64>> = match model {
        Model::Linear => vec![None],
        Model::Power => (0..=100)
            .map(|k| Some(0.001 + 1.998 * k as f64 / 100.0))
            .collect(),
        _ => (1..=80).map(|k| Some(max_lag * k as f64 / 80.0)).collect(),
    };
    let mut candidates: Vec<(Vec<f64>, f64)> = nonlinear
        .iter()
        .map(|&nl| {
            let p = robust_linear(model, nl, lags, sv, &w, &lb, &ub);
            let c = objective(model, &p, lags, sv, weight);
            (p, c)
        })
        .collect();
    candidates.sort_by(|a, b| a.1.total_cmp(&b.1));
    let scale: Vec<f64> = ub
        .iter()
        .zip([max_sv, max_lag, max_sv])
        .map(|(u, fallback)| {
            if u.is_finite() {
                u.abs().max(1e-12)
            } else {
                fallback.abs().max(1.0)
            }
        })
        .collect();
    // PyKrige's own starting point joins the best scan points.
    let min_sv = sv.iter().copied().fold(f64::INFINITY, f64::min);
    let min_lag = lags.iter().copied().fold(f64::INFINITY, f64::min);
    let pykrige_x0 = match model {
        Model::Linear => vec![(max_sv - min_sv) / (max_lag - min_lag), min_sv],
        Model::Power => vec![(max_sv - min_sv) / (max_lag - min_lag), 1.1, min_sv],
        _ => vec![max_sv - min_sv, 0.25 * max_lag, min_sv],
    };
    let mut starts: Vec<Vec<f64>> = candidates.into_iter().take(6).map(|(p, _)| p).collect();
    starts.push(pykrige_x0);
    let best = starts
        .into_iter()
        .map(|p| {
            // Restart from the result until it stops improving: one run may stop early on a
            // narrow valley.
            let (mut p, mut c) = minimise(&transformed, p, &lb, &ub, &scale);
            for _ in 0..20 {
                let (q, d) = minimise(&transformed, p.clone(), &lb, &ub, &scale);
                if d >= c * (1.0 - 1e-15) {
                    break;
                }
                (p, c) = (q, d);
            }
            (p, c)
        })
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(p, _)| p)
        .ok_or("the automatic variogram fit failed")?;
    Ok(best)
}

/// Robust (soft_l1) fit of the multiplier and the nugget for a fixed non-linear parameter.
fn robust_linear(
    model: Model,
    nonlinear: Option<f64>,
    lags: &[f64],
    sv: &[f64],
    w: &[f64],
    lb: &[f64],
    ub: &[f64],
) -> Vec<f64> {
    // Basis g(l): the model with multiplier 1 and nugget 0.
    let assemble = |mult: f64, nugget: f64| -> Vec<f64> {
        match (model, nonlinear) {
            (Model::Linear, _) => vec![mult, nugget],
            (Model::Power, Some(e)) => vec![mult, e, nugget],
            (_, Some(r)) => vec![mult, r, nugget],
            _ => vec![mult, nugget],
        }
    };
    let g: Vec<f64> = lags
        .iter()
        .map(|&l| value(model, &assemble(1.0, 0.0), l))
        .collect();
    let (mult_lo, mult_hi) = (lb[0], ub[0]);
    let (nug_lo, nug_hi) = (lb[lb.len() - 1], ub[ub.len() - 1]);
    let mut mult = 0.0;
    let mut nugget = 0.0;
    for _ in 0..60 {
        // IRLS weights of soft_l1: rho'(r^2) = 1 / sqrt(1 + r^2), times the lag weight squared.
        let iw: Vec<f64> = (0..lags.len())
            .map(|i| {
                let r = (mult * g[i] + nugget - sv[i]) * w[i];
                w[i] * w[i] / (1.0 + r * r).sqrt()
            })
            .collect();
        let (next_m, next_n) = bounded_line(&g, sv, &iw, (mult_lo, mult_hi), (nug_lo, nug_hi));
        let done = (next_m - mult).abs() <= 1e-14 * next_m.abs().max(1.0)
            && (next_n - nugget).abs() <= 1e-14 * next_n.abs().max(1.0);
        mult = next_m;
        nugget = next_n;
        if done {
            break;
        }
    }
    assemble(mult, nugget)
}

/// Weighted least squares of `y ~ m g + b` with `m` and `b` in their intervals.
fn bounded_line(
    g: &[f64],
    y: &[f64],
    w: &[f64],
    m_box: (f64, f64),
    b_box: (f64, f64),
) -> (f64, f64) {
    let sse = |m: f64, b: f64| -> f64 {
        (0..g.len())
            .map(|i| w[i] * (m * g[i] + b - y[i]).powi(2))
            .sum()
    };
    let s = |f: &dyn Fn(usize) -> f64| (0..g.len()).map(f).sum::<f64>();
    let (sw, sg, sgg, sy, sgy) = (
        s(&|i| w[i]),
        s(&|i| w[i] * g[i]),
        s(&|i| w[i] * g[i] * g[i]),
        s(&|i| w[i] * y[i]),
        s(&|i| w[i] * g[i] * y[i]),
    );
    let clamp_m = |m: f64| m.clamp(m_box.0, m_box.1);
    let clamp_b = |b: f64| b.clamp(b_box.0, b_box.1);
    let mut cands: Vec<(f64, f64)> = Vec::new();
    if let Some(sol) = solve_small(vec![vec![sgg, sg], vec![sg, sw]], vec![sgy, sy]) {
        cands.push((clamp_m(sol[0]), clamp_b(sol[1])));
    }
    // Each bound active: the other parameter optimal given it.
    for b in [b_box.0, b_box.1].into_iter().filter(|v| v.is_finite()) {
        let m = if sgg > 0.0 { (sgy - b * sg) / sgg } else { 0.0 };
        cands.push((clamp_m(m), b));
    }
    for m in [m_box.0, m_box.1].into_iter().filter(|v| v.is_finite()) {
        let b = if sw > 0.0 { (sy - m * sg) / sw } else { 0.0 };
        cands.push((m, clamp_b(b)));
    }
    cands
        .into_iter()
        .min_by(|a, b| sse(a.0, a.1).total_cmp(&sse(b.0, b.1)))
        .unwrap_or((0.0, 0.0))
}

/// The inverse (or pseudo-inverse) of a kriging matrix.
fn invert(a: &Mat<f64>, inverse: Inverse) -> Result<Mat<f64>, String> {
    let n = a.nrows();
    let eps = f64::EPSILON;
    match inverse {
        Inverse::Exact => {
            let lu = a.partial_piv_lu();
            let inv = lu.inverse();
            if (0..n).any(|i| (0..n).any(|j| !inv[(i, j)].is_finite())) {
                return Err("the kriging matrix is singular (try pseudo_inv: true)".to_string());
            }
            Ok(inv)
        }
        Inverse::Pinv => {
            let svd = a.thin_svd().map_err(|e| format!("SVD failed: {e:?}"))?;
            let (u, s, v) = (svd.U(), svd.S(), svd.V());
            let s: Vec<f64> = (0..s.dim()).map(|i| s[i]).collect();
            let cutoff = n as f64 * eps * s.iter().copied().fold(0.0, f64::max);
            Ok(Mat::from_fn(n, n, |i, j| {
                (0..s.len())
                    .filter(|&k| s[k] > cutoff)
                    .map(|k| v[(i, k)] * u[(j, k)] / s[k])
                    .sum()
            }))
        }
        Inverse::Pinvh => {
            let eig = a
                .self_adjoint_eigen(Side::Lower)
                .map_err(|e| format!("eigendecomposition failed: {e:?}"))?;
            let (u, s) = (eig.U(), eig.S());
            let w: Vec<f64> = (0..n).map(|i| s[i]).collect();
            let cutoff = n as f64 * eps * w.iter().map(|v| v.abs()).fold(0.0, f64::max);
            Ok(Mat::from_fn(n, n, |i, j| {
                (0..n)
                    .filter(|&k| w[k].abs() > cutoff)
                    .map(|k| u[(i, k)] * u[(j, k)] / w[k])
                    .sum()
            }))
        }
    }
}

impl Kriging {
    fn gamma(&self, d: f64) -> f64 {
        match &self.gamma_table {
            Some(table) => table.get(&d.to_bits()).copied().unwrap_or(f64::NAN),
            None => value(self.model, &self.params, d),
        }
    }

    /// Every distance a custom variogram is evaluated at to krige `targets` (the unmasked ones;
    /// none for the statistics): between the data points in both orders and from each target to
    /// each data point, computed as kriging computes them. Sorted and without repeats. An error
    /// when the table would not fit in memory.
    pub fn distances(&self, targets: &[Point], masked: &[bool]) -> Result<Vec<f64>, String> {
        let n = self.points.len();
        // The table, the stable sort's buffer and the copy handed to Elixir.
        memory::ensure(20.0 * n as f64 * (n + targets.len()) as f64, || {
            format!(
                "the custom variogram table of {n} points and {} targets",
                targets.len()
            )
        })?;
        let mut out: Vec<f64> = Vec::with_capacity(n * n);
        for (i, &a) in self.points.iter().enumerate() {
            for (j, &b) in self.points.iter().enumerate() {
                if i != j {
                    out.push(dist(a, b, self.geographic));
                }
            }
        }
        for (t, &p) in targets.iter().enumerate() {
            if !masked.get(t).copied().unwrap_or(false) {
                out.extend(self.points.iter().map(|&q| dist(p, q, self.geographic)));
            }
        }
        out.sort_by(f64::total_cmp);
        out.dedup_by(|a, b| a.to_bits() == b.to_bits());
        Ok(out)
    }

    /// The kriging matrix of the selected data points: the semivariances, the drift columns and
    /// the unbiasedness row (`(n + d + 1)` square, `d` drift columns).
    fn matrix(&self, idx: &[usize]) -> Mat<f64> {
        let n = idx.len();
        let d = self.data_drift.len();
        let size = n + d + 1;
        Mat::from_fn(size, size, |i, j| {
            if i < n && j < n {
                if i == j {
                    0.0
                } else {
                    -self.gamma(dist(
                        self.points[idx[i]],
                        self.points[idx[j]],
                        self.geographic,
                    ))
                }
            } else if i < n && j < n + d {
                self.data_drift[j - n][idx[i]]
            } else if j < n && i < n + d {
                self.data_drift[i - n][idx[j]]
            } else if (i < n && j == n + d) || (j < n && i == n + d) {
                1.0
            } else {
                0.0
            }
        })
    }

    fn invert(&self, a: &Mat<f64>) -> Result<Mat<f64>, String> {
        invert(a, self.inverse)
    }

    /// PyKrige's `_find_statistics`: every point kriged (ordinary kriging, without drift) from
    /// the points before it. Returns the residuals `z - prediction` and the kriging standard
    /// deviations of the points whose kriging variance is not about 0. A pseudo-inverse model
    /// solves by least squares, as PyKrige does. The points are kriged in parallel (each system
    /// on its own, as one after another) and collected in order.
    pub fn statistics(&self) -> Result<(Vec<f64>, Vec<f64>), String> {
        let n = self.points.len();
        // One system per thread at a time, as many threads as the largest systems fit in memory.
        let threads = memory::fitting(
            memory::system_bytes(n, self.inverse != Inverse::Exact),
            || format!("the statistics of {n} points"),
        )?;
        let rows = par_map_dynamic(n.saturating_sub(1), threads, |j| self.statistic(j + 1));
        let (mut delta, mut sigma) = (Vec::new(), Vec::new());
        for row in rows {
            if let Some((d, s)) = row? {
                delta.push(d);
                sigma.push(s);
            }
        }
        Ok((delta, sigma))
    }

    /// Point `i` (at least 1) kriged from the points before it: its residual and kriging
    /// standard deviation, or `None` when its kriging variance is about 0.
    fn statistic(&self, i: usize) -> Result<Option<(f64, f64)>, String> {
        const EPS: f64 = 1e-10;
        let a = Mat::from_fn(i + 1, i + 1, |r, c| {
            if r == c {
                0.0
            } else if r < i && c < i {
                -self.gamma(dist(self.points[r], self.points[c], self.geographic))
            } else {
                1.0
            }
        });
        let d: Vec<f64> = (0..i)
            .map(|k| dist(self.points[i], self.points[k], self.geographic))
            .collect();
        let zero = d.iter().position(|v| v.abs() <= EPS);
        let mut b: Vec<f64> = d
            .iter()
            .enumerate()
            .map(|(k, &v)| if Some(k) == zero { 0.0 } else { -self.gamma(v) })
            .collect();
        b.push(1.0);
        let x: Vec<f64> = if self.inverse == Inverse::Exact {
            let rhs = Mat::from_fn(i + 1, 1, |r, _| b[r]);
            let sol = a.partial_piv_lu().solve(&rhs);
            (0..=i).map(|r| sol[(r, 0)]).collect()
        } else {
            let inv = invert(&a, Inverse::Pinv)?;
            (0..=i)
                .map(|r| (0..=i).map(|c| inv[(r, c)] * b[c]).sum())
                .collect()
        };
        let z: f64 = (0..i).map(|k| x[k] * self.values[k]).sum();
        let ss: f64 = (0..=i).map(|k| x[k] * -b[k]).sum();
        if ss.abs() < EPS {
            return Ok(None);
        }
        let s = ss.sqrt();
        Ok((s > EPS).then(|| (self.values[i] - z, s)))
    }

    /// The sill of a model that has one (partial sill plus nugget).
    fn sill(&self) -> f64 {
        self.params[0] + self.params[2]
    }

    /// Simple kriging's matrix (`gstools.krige.Simple`): the covariance `sill - gamma` between
    /// the data points, with the sill (the covariance plus the nugget as measurement error) on
    /// the diagonal.
    fn covariance_matrix(&self) -> Mat<f64> {
        let (n, sill) = (self.points.len(), self.sill());
        Mat::from_fn(n, n, |i, j| {
            if i == j {
                sill
            } else {
                sill - self.gamma(dist(self.points[i], self.points[j], self.geographic))
            }
        })
    }

    /// A simple-kriging prediction: `mean + w . (z - mean)` and the variance
    /// `max(sill - w . c, 0)`, `w = A^-1 c`; with exact values the covariance at distance 0
    /// includes the nugget, so the data are reproduced.
    fn predict_simple(&self, a_inv: &Mat<f64>, mean: f64, p: Point) -> (f64, f64) {
        let sill = self.sill();
        let c: Vec<f64> = self
            .points
            .iter()
            .map(|&q| {
                let d = dist(p, q, self.geographic);
                if d.abs() <= 1e-10 && self.exact_values {
                    sill
                } else {
                    sill - self.gamma(d)
                }
            })
            .collect();
        let w = times(a_inv, &c);
        let z = mean
            + w.iter()
                .zip(&self.values)
                .map(|(wk, v)| wk * (v - mean))
                .sum::<f64>();
        let s = sill - w.iter().zip(&c).map(|(wk, ck)| wk * ck).sum::<f64>();
        (z, s.max(0.0))
    }

    fn predict_with(&self, a_inv: &Mat<f64>, idx: &[usize], p: Point, drift: &[f64]) -> (f64, f64) {
        let n = idx.len();
        let size = n + drift.len() + 1;
        let mut b = vec![0.0; size];
        for (k, &i) in idx.iter().enumerate() {
            let d = dist(p, self.points[i], self.geographic);
            b[k] = if d.abs() <= 1e-10 && self.exact_values {
                0.0
            } else {
                -self.gamma(d)
            };
        }
        b[n..n + drift.len()].copy_from_slice(drift);
        b[size - 1] = 1.0;
        let x = times(a_inv, &b);
        let z = (0..n).map(|k| x[k] * self.values[idx[k]]).sum();
        let s = (0..size).map(|k| x[k] * -b[k]).sum();
        (z, s)
    }

    /// Ordinary block kriging (`gstat`'s `block =`): each target is the centre of a block
    /// discretised by `block` (`[dx, dy, weight]` offsets, already anisotropy-adjusted, weights
    /// summing to 1). The right-hand side is the weighted average of the point semivariances, the
    /// estimate the block's mean, and the variance subtracts the block-to-block average
    /// `sum_pq w_p w_q gamma(o_p - o_q)`. As gstat, the semivariance at distance 0 is the
    /// nugget inside a block (a block averages the field without its measurement error), both
    /// in that average and where a discretisation point falls on a datum. Global system, no
    /// drift.
    pub fn execute_block(
        &self,
        targets: &[Point],
        masked: &[bool],
        block: &[[f64; 3]],
    ) -> Result<(Vec<f64>, Vec<f64>), String> {
        if !self.data_drift.is_empty() || self.simple_mean.is_some() {
            return Err("block kriging is available for ordinary kriging only".to_string());
        }
        if block.is_empty() {
            return Err("a block needs at least one discretisation point".to_string());
        }
        let n = self.points.len();
        memory::ensure(
            memory::system_bytes(n + 1, self.inverse != Inverse::Exact),
            || format!("block kriging of {n} points"),
        )?;
        let all: Vec<usize> = (0..n).collect();
        let a_inv = self.invert(&self.matrix(&all))?;
        let at = |o: &[f64; 3]| [o[0], o[1], 0.0];
        let block_variance: f64 = block
            .iter()
            .map(|p| {
                block
                    .iter()
                    .map(|q| p[2] * q[2] * self.gamma(dist(at(p), at(q), self.geographic)))
                    .sum::<f64>()
            })
            .sum();
        let out = par_map(targets.len(), MIN_TARGETS_PER_THREAD, |t| {
            if masked[t] {
                return (f64::NAN, f64::NAN);
            }
            let p = targets[t];
            let mut b: Vec<f64> = self
                .points
                .iter()
                .map(|&q| {
                    -block
                        .iter()
                        .map(|o| {
                            let d = dist([p[0] + o[0], p[1] + o[1], p[2]], q, self.geographic);
                            o[2] * self.gamma(d)
                        })
                        .sum::<f64>()
                })
                .collect();
            b.push(1.0);
            let x = times(&a_inv, &b);
            let z = (0..n).map(|k| x[k] * self.values[k]).sum();
            let s = (0..=n).map(|k| x[k] * -b[k]).sum::<f64>() - block_variance;
            (z, s)
        });
        Ok(out.into_iter().unzip())
    }

    /// Predictions and kriging variances at `targets` (already anisotropy-adjusted); `None`
    /// entries of `mask` are skipped (`masked` is true). With `n_closest`, each prediction uses
    /// only its nearest data points.
    pub fn execute(
        &self,
        targets: &[Point],
        target_drift: &[Vec<f64>],
        masked: &[bool],
        n_closest: Option<usize>,
    ) -> Result<(Vec<f64>, Vec<f64>), String> {
        let n = self.points.len();
        if target_drift.len() != self.data_drift.len() {
            return Err("the drift terms at the targets do not match the model".to_string());
        }
        if n_closest.is_some() && !self.data_drift.is_empty() {
            return Err("n_closest_points is not available with drift terms".to_string());
        }
        if let Some(mean) = self.simple_mean {
            if n_closest.is_some() {
                return Err("n_closest_points is not available with simple kriging".to_string());
            }
            memory::ensure(
                memory::system_bytes(n, self.inverse != Inverse::Exact),
                || format!("simple kriging of {n} points"),
            )?;
            let a_inv = self.invert(&self.covariance_matrix())?;
            let out = par_map(targets.len(), MIN_TARGETS_PER_THREAD, |t| {
                if masked[t] {
                    return (f64::NAN, f64::NAN);
                }
                self.predict_simple(&a_inv, mean, targets[t])
            });
            return Ok(out.into_iter().unzip());
        }
        let out = match n_closest {
            None => {
                let size = n + self.data_drift.len() + 1;
                memory::ensure(
                    memory::system_bytes(size, self.inverse != Inverse::Exact),
                    || format!("global kriging of {n} points"),
                )
                .map_err(|e| {
                    format!("{e}; use n_closest_points (a moving window) or fewer points")
                })?;
                let all: Vec<usize> = (0..n).collect();
                let a_inv = self.invert(&self.matrix(&all))?;
                par_map(targets.len(), MIN_TARGETS_PER_THREAD, |t| {
                    if masked[t] {
                        return (f64::NAN, f64::NAN);
                    }
                    let drift: Vec<f64> = target_drift.iter().map(|c| c[t]).collect();
                    self.predict_with(&a_inv, &all, targets[t], &drift)
                })
            }
            Some(k) => {
                let k = k.min(n);
                let tree = KdTree::new(if self.geographic {
                    self.points.iter().map(|&q| unit_vector(q)).collect()
                } else {
                    self.points.clone()
                });
                par_map(targets.len(), MIN_TARGETS_PER_THREAD, |t| {
                    if masked[t] {
                        return (f64::NAN, f64::NAN);
                    }
                    self.predict_window(targets[t], k, tree.as_ref())
                })
            }
        };
        Ok(out.into_iter().unzip())
    }

    /// One moving-window prediction from the `k` nearest data points (by `selection_distance`,
    /// ties by index), solved directly (`scipy.linalg.solve`).
    fn predict_window(&self, p: Point, k: usize, tree: Option<&KdTree>) -> (f64, f64) {
        let idx: Vec<usize> = match tree {
            Some(tree) => {
                let q = if self.geographic { unit_vector(p) } else { p };
                tree.nearest(q, k).into_iter().map(|(_, i)| i).collect()
            }
            None => {
                let mut order: Vec<(f64, usize)> = (0..self.points.len())
                    .map(|i| (self.selection_distance(p, i), i))
                    .collect();
                order.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                order.into_iter().take(k).map(|(_, i)| i).collect()
            }
        };
        let a = self.matrix(&idx);
        let lu = a.partial_piv_lu();
        let b = Mat::from_fn(k + 1, 1, |r, _| {
            if r == k {
                1.0
            } else {
                let d = dist(p, self.points[idx[r]], self.geographic);
                if d.abs() <= 1e-10 && self.exact_values {
                    0.0
                } else {
                    -self.gamma(d)
                }
            }
        });
        let x = lu.solve(&b);
        let z = (0..k).map(|r| x[(r, 0)] * self.values[idx[r]]).sum();
        let s = -(0..=k).map(|r| x[(r, 0)] * b[(r, 0)]).sum::<f64>();
        (z, s)
    }

    /// Distance used to pick the nearest data points: Euclidean, or the chord between unit
    /// vectors for geographic coordinates (PyKrige's KD-tree on the sphere).
    fn selection_distance(&self, p: Point, i: usize) -> f64 {
        if self.geographic {
            let (a, b) = (unit_vector(p), unit_vector(self.points[i]));
            ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
        } else {
            dist(p, self.points[i], false)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line_data() -> Kriging {
        Kriging {
            points: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ],
            values: vec![1.0, 2.0, 4.0, 3.0],
            model: Model::Linear,
            params: vec![1.0, 0.0],
            geographic: false,
            exact_values: true,
            inverse: Inverse::Exact,
            center: [1.5, 0.0, 0.0],
            scaling: [1.0, 1.0],
            angles: [0.0; 3],
            drifts: Vec::new(),
            data_drift: Vec::new(),
            gamma_table: None,
            simple_mean: None,
        }
    }

    #[test]
    fn a_gamma_table_replaces_the_model() {
        // A custom variogram given as a table of the distances it needs reproduces the model.
        let linear = line_data();
        let targets = [[0.5, 0.0, 0.0], [2.0, 0.0, 0.0], [5.0, 1.0, 0.0]];
        let masked = [false, true, false];
        let d = linear.distances(&targets, &masked).unwrap();
        assert!(d.windows(2).all(|w| w[0] < w[1]), "sorted and unique");
        // Data-data distances 1, 2, 3 and the unmasked targets' distances are all there.
        for needed in [1.0, 2.0, 3.0, 0.5, 2.5, (25.0_f64 + 1.0).sqrt()] {
            assert!(d.contains(&needed), "{needed} missing from {d:?}");
        }
        assert!(
            !d.contains(&0.0),
            "the masked target at a data point is skipped"
        );
        let mut custom = line_data();
        custom.model = Model::Custom;
        custom.params = vec![42.0];
        custom.gamma_table = Some(d.iter().map(|&x| (x.to_bits(), x)).collect());
        let expected = linear.execute(&targets, &[], &masked, None).unwrap();
        let got = custom.execute(&targets, &[], &masked, None).unwrap();
        for (a, b) in got
            .0
            .iter()
            .zip(&expected.0)
            .chain(got.1.iter().zip(&expected.1))
        {
            assert!(
                a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()),
                "{a} vs {b}"
            );
        }
        assert_eq!(custom.statistics().unwrap(), linear.statistics().unwrap());
        assert!(fit(Model::Custom, &[1.0, 2.0], &[1.0, 2.0], false).is_err());
    }

    #[test]
    fn kriging_is_exact_at_data_points_and_weights_sum_to_one() {
        let ok = line_data();
        let (z, s) = ok
            .execute(
                &[[1.0, 0.0, 0.0], [1.5, 0.0, 0.0]],
                &[],
                &[false, false],
                None,
            )
            .unwrap();
        assert!((z[0] - 2.0).abs() < 1e-12 && s[0].abs() < 1e-12);
        // Midway on a linear variogram: the average of the two neighbours.
        assert!((z[1] - 3.0).abs() < 1e-12);
        let (zw, _) = ok
            .execute(&[[1.5, 0.0, 0.0]], &[], &[false], Some(4))
            .unwrap();
        assert!((zw[0] - z[1]).abs() < 1e-12);
    }

    #[test]
    fn a_regional_drift_reproduces_a_plane_exactly() {
        let points = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 0.5, 0.0],
        ];
        let values: Vec<f64> = points.iter().map(|p| 1.0 + 2.0 * p[0] - p[1]).collect();
        let drifts = vec![Drift::RegionalLinear];
        let data_drift = drift_columns(&drifts, &points, &points).unwrap();
        let uk = Kriging {
            points,
            values,
            drifts,
            data_drift,
            ..line_data()
        };
        let target = [[0.4, 0.7, 0.0]];
        let td = drift_columns(&uk.drifts, &target, &target).unwrap();
        let (z, _) = uk.execute(&target, &td, &[false], None).unwrap();
        assert!((z[0] - (1.0 + 0.8 - 0.7)).abs() < 1e-10, "{}", z[0]);
    }

    #[test]
    fn great_circle_quarter_and_anisotropy_rotation() {
        assert!((great_circle(0.0, 0.0, 90.0, 0.0) - 90.0).abs() < 1e-12);
        let p = adjust(&[[1.0, 0.0, 0.0]], [0.0; 3], [2.0, 1.0], [0.0, 0.0, 90.0]);
        assert!((p[0][0]).abs() < 1e-12 && (p[0][1] + 2.0).abs() < 1e-12);
    }

    #[test]
    fn pykrige_lags_use_the_mean_distance_of_each_lag() {
        let pts = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]];
        let (l, s) = lags(&pts, &[0.0, 1.0, 3.0], 2, false);
        // Distances 1, 3, 2; lags [1, 2) and [2, 3.001).
        assert_eq!(l, vec![1.0, 2.5]);
        assert_eq!(s, vec![0.5, (4.5 + 2.0) / 2.0]);
    }

    /// A small deterministic generator for test data (xorshift64*), in [0, 1).
    pub(crate) fn uniform(state: &mut u64) -> f64 {
        *state ^= *state >> 12;
        *state ^= *state << 25;
        *state ^= *state >> 27;
        (state.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 11) as f64 / (1u64 << 53) as f64
    }

    /// The pair-list binning `lags` replaced (every pair stored, one scan per lag): the
    /// reference the one-pass version must match bit for bit.
    fn naive_lags(
        points: &[Point],
        values: &[f64],
        nlags: usize,
        geographic: bool,
    ) -> (Vec<f64>, Vec<f64>) {
        let n = points.len();
        let (mut d, mut g) = (Vec::new(), Vec::new());
        for i in 0..n {
            for j in (i + 1)..n {
                d.push(dist(points[i], points[j], geographic));
                g.push(0.5 * (values[i] - values[j]).powi(2));
            }
        }
        let dmax = d.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let dmin = d.iter().copied().fold(f64::INFINITY, f64::min);
        let dd = (dmax - dmin) / nlags as f64;
        let mut bins: Vec<f64> = (0..nlags).map(|k| dmin + k as f64 * dd).collect();
        bins.push(dmax + 0.001);
        let (mut out_lags, mut out_sv) = (Vec::new(), Vec::new());
        for k in 0..nlags {
            let (mut sd, mut sg, mut cnt) = (0.0, 0.0, 0usize);
            for (dv, gv) in d.iter().zip(&g) {
                if *dv >= bins[k] && *dv < bins[k + 1] {
                    sd += dv;
                    sg += gv;
                    cnt += 1;
                }
            }
            if cnt > 0 {
                out_lags.push(sd / cnt as f64);
                out_sv.push(sg / cnt as f64);
            }
        }
        (out_lags, out_sv)
    }

    fn bits(v: &[f64]) -> Vec<u64> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    #[test]
    fn lags_match_the_pair_list_binning_bit_for_bit() {
        let mut s = 0x9e37_79b9_7f4a_7c15;
        let mut cases: Vec<(Vec<Point>, bool)> = Vec::new();
        // Scattered points, planar and geographic (degrees).
        let scattered: Vec<Point> = (0..300)
            .map(|_| [uniform(&mut s) * 1000.0, uniform(&mut s) * 1000.0, 0.0])
            .collect();
        cases.push((scattered, false));
        let geo: Vec<Point> = (0..200)
            .map(|_| {
                [
                    uniform(&mut s) * 40.0 - 20.0,
                    uniform(&mut s) * 30.0 + 30.0,
                    0.0,
                ]
            })
            .collect();
        cases.push((geo, true));
        // A regular grid: many equal distances, some exactly on lag edges.
        let grid: Vec<Point> = (0..15)
            .flat_map(|i| (0..15).map(move |j| [i as f64, j as f64, 0.0]))
            .collect();
        cases.push((grid, false));
        // 3D points, a duplicate point (distance 0) and a single distance (dmin = dmax).
        let cube: Vec<Point> = (0..150)
            .map(|_| [uniform(&mut s), uniform(&mut s), uniform(&mut s)])
            .collect();
        cases.push((cube, false));
        cases.push((
            vec![[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
            false,
        ));
        cases.push((vec![[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]], false));
        for (points, geographic) in cases {
            let values: Vec<f64> = points
                .iter()
                .map(|_| uniform(&mut s) * 10.0 - 5.0)
                .collect();
            for nlags in [1, 2, 6, 12, 40] {
                let (l, v) = lags(&points, &values, nlags, geographic);
                let (nl, nv) = naive_lags(&points, &values, nlags, geographic);
                assert_eq!(
                    bits(&l),
                    bits(&nl),
                    "lags, n {} nlags {nlags}",
                    points.len()
                );
                assert_eq!(
                    bits(&v),
                    bits(&nv),
                    "semivariances, n {} nlags {nlags}",
                    points.len()
                );
            }
        }
    }

    /// `execute` as it was before the prediction work (one target at a time, a full sort for the
    /// moving window, the inverse read row by row): the reference the new one must match bit for
    /// bit.
    fn reference_execute(
        k: &Kriging,
        targets: &[Point],
        target_drift: &[Vec<f64>],
        masked: &[bool],
        n_closest: Option<usize>,
    ) -> (Vec<f64>, Vec<f64>) {
        let n = k.points.len();
        let mut z = vec![f64::NAN; targets.len()];
        let mut s = vec![f64::NAN; targets.len()];
        match n_closest {
            None => {
                let all: Vec<usize> = (0..n).collect();
                let a_inv = k.invert(&k.matrix(&all)).unwrap();
                for (t, p) in targets.iter().enumerate() {
                    if masked[t] {
                        continue;
                    }
                    let drift: Vec<f64> = target_drift.iter().map(|c| c[t]).collect();
                    let size = n + drift.len() + 1;
                    let mut b = vec![0.0; size];
                    for (j, &i) in all.iter().enumerate() {
                        let d = dist(*p, k.points[i], k.geographic);
                        b[j] = if d.abs() <= 1e-10 && k.exact_values {
                            0.0
                        } else {
                            -k.gamma(d)
                        };
                    }
                    b[n..n + drift.len()].copy_from_slice(&drift);
                    b[size - 1] = 1.0;
                    let x: Vec<f64> = (0..size)
                        .map(|r| (0..size).map(|c| a_inv[(r, c)] * b[c]).sum())
                        .collect();
                    z[t] = (0..n).map(|j| x[j] * k.values[all[j]]).sum();
                    s[t] = (0..size).map(|j| x[j] * -b[j]).sum();
                }
            }
            Some(kk) => {
                let kk = kk.min(n);
                for (t, p) in targets.iter().enumerate() {
                    if masked[t] {
                        continue;
                    }
                    let mut order: Vec<(f64, usize)> =
                        (0..n).map(|i| (k.selection_distance(*p, i), i)).collect();
                    order.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                    let idx: Vec<usize> = order.into_iter().take(kk).map(|(_, i)| i).collect();
                    let a = k.matrix(&idx);
                    let lu = a.partial_piv_lu();
                    let b = Mat::from_fn(kk + 1, 1, |r, _| {
                        if r == kk {
                            1.0
                        } else {
                            let d = dist(*p, k.points[idx[r]], k.geographic);
                            if d.abs() <= 1e-10 && k.exact_values {
                                0.0
                            } else {
                                -k.gamma(d)
                            }
                        }
                    });
                    let x = lu.solve(&b);
                    z[t] = (0..kk).map(|r| x[(r, 0)] * k.values[idx[r]]).sum();
                    s[t] = -(0..=kk).map(|r| x[(r, 0)] * b[(r, 0)]).sum::<f64>();
                }
            }
        }
        (z, s)
    }

    fn model_over(points: Vec<Point>, s: &mut u64, model: Model, params: Vec<f64>) -> Kriging {
        let values = points.iter().map(|_| uniform(s) * 4.0 - 2.0).collect();
        Kriging {
            points,
            values,
            model,
            params,
            ..line_data()
        }
    }

    #[test]
    fn execute_matches_the_sequential_reference_bit_for_bit() {
        let mut s = 0x2545_f491_4f6c_dd1d;
        let mut models: Vec<Kriging> = Vec::new();
        let scattered: Vec<Point> = (0..120)
            .map(|_| [uniform(&mut s) * 100.0, uniform(&mut s) * 100.0, 0.0])
            .collect();
        models.push(model_over(
            scattered.clone(),
            &mut s,
            Model::Spherical,
            vec![1.5, 40.0, 0.1],
        ));
        // A regular grid: many equal distances, so the window's tie order matters.
        let grid: Vec<Point> = (0..11)
            .flat_map(|i| (0..11).map(move |j| [i as f64 * 5.0, j as f64 * 5.0, 0.0]))
            .collect();
        models.push(model_over(
            grid,
            &mut s,
            Model::Exponential,
            vec![1.0, 30.0, 0.0],
        ));
        let mut geo = model_over(
            (0..100)
                .map(|_| {
                    [
                        uniform(&mut s) * 40.0 - 20.0,
                        uniform(&mut s) * 30.0 + 30.0,
                        0.0,
                    ]
                })
                .collect(),
            &mut s,
            Model::Gaussian,
            vec![1.0, 15.0, 0.05],
        );
        geo.geographic = true;
        models.push(geo);
        let cube: Vec<Point> = (0..100)
            .map(|_| {
                [
                    uniform(&mut s) * 10.0,
                    uniform(&mut s) * 10.0,
                    uniform(&mut s) * 10.0,
                ]
            })
            .collect();
        models.push(model_over(cube, &mut s, Model::Linear, vec![0.3, 0.02]));
        let mut pinv = model_over(scattered, &mut s, Model::Spherical, vec![1.5, 40.0, 0.1]);
        pinv.inverse = Inverse::Pinv;
        models.push(pinv);

        for (m, k) in models.iter().enumerate() {
            let (lo, hi) = if k.geographic {
                ([-20.0, 30.0], [20.0, 60.0])
            } else {
                ([0.0, 0.0], [100.0, 100.0])
            };
            let three_d = k.points.iter().any(|p| p[2] != 0.0);
            let mut targets: Vec<Point> = (0..80)
                .map(|_| {
                    [
                        lo[0] + uniform(&mut s) * (hi[0] - lo[0]),
                        lo[1] + uniform(&mut s) * (hi[1] - lo[1]),
                        if three_d { uniform(&mut s) * 10.0 } else { 0.0 },
                    ]
                })
                .collect();
            // Some targets exactly at data points (exact values) and some masked.
            targets.extend(k.points.iter().take(7).copied());
            let masked: Vec<bool> = (0..targets.len()).map(|t| t % 11 == 3).collect();
            for n_closest in [None, Some(1), Some(8), Some(25), Some(k.points.len() + 5)] {
                let (z, v) = k.execute(&targets, &[], &masked, n_closest).unwrap();
                let (rz, rv) = reference_execute(k, &targets, &[], &masked, n_closest);
                assert_eq!(
                    bits(&z),
                    bits(&rz),
                    "model {m}, n_closest {n_closest:?}: estimates"
                );
                assert_eq!(
                    bits(&v),
                    bits(&rv),
                    "model {m}, n_closest {n_closest:?}: variances"
                );
            }
        }

        // Universal kriging (two drift columns), global only.
        let mut uk = model_over(
            (0..80)
                .map(|_| [uniform(&mut s) * 50.0, uniform(&mut s) * 50.0, 0.0])
                .collect(),
            &mut s,
            Model::Linear,
            vec![0.2, 0.0],
        );
        uk.data_drift = vec![
            uk.points.iter().map(|p| p[0]).collect(),
            uk.points.iter().map(|p| p[1]).collect(),
        ];
        let targets: Vec<Point> = (0..60)
            .map(|_| [uniform(&mut s) * 50.0, uniform(&mut s) * 50.0, 0.0])
            .collect();
        let drift = vec![
            targets.iter().map(|p| p[0]).collect::<Vec<f64>>(),
            targets.iter().map(|p| p[1]).collect::<Vec<f64>>(),
        ];
        let masked = vec![false; targets.len()];
        let (z, v) = uk.execute(&targets, &drift, &masked, None).unwrap();
        let (rz, rv) = reference_execute(&uk, &targets, &drift, &masked, None);
        assert_eq!(bits(&z), bits(&rz));
        assert_eq!(bits(&v), bits(&rv));
    }

    /// `statistics` as it was before it ran in parallel (one point after another): the reference
    /// the new one must match bit for bit.
    fn reference_statistics(k: &Kriging) -> Result<(Vec<f64>, Vec<f64>), String> {
        const EPS: f64 = 1e-10;
        let n = k.points.len();
        let (mut delta, mut sigma) = (Vec::new(), Vec::new());
        for i in 1..n {
            let a = Mat::from_fn(i + 1, i + 1, |r, c| {
                if r == c {
                    0.0
                } else if r < i && c < i {
                    -k.gamma(dist(k.points[r], k.points[c], k.geographic))
                } else {
                    1.0
                }
            });
            let d: Vec<f64> = (0..i)
                .map(|j| dist(k.points[i], k.points[j], k.geographic))
                .collect();
            let zero = d.iter().position(|v| v.abs() <= EPS);
            let mut b: Vec<f64> = d
                .iter()
                .enumerate()
                .map(|(j, &v)| if Some(j) == zero { 0.0 } else { -k.gamma(v) })
                .collect();
            b.push(1.0);
            let x: Vec<f64> = if k.inverse == Inverse::Exact {
                let rhs = Mat::from_fn(i + 1, 1, |r, _| b[r]);
                let sol = a.partial_piv_lu().solve(&rhs);
                (0..=i).map(|r| sol[(r, 0)]).collect()
            } else {
                let inv = invert(&a, Inverse::Pinv)?;
                (0..=i)
                    .map(|r| (0..=i).map(|c| inv[(r, c)] * b[c]).sum())
                    .collect()
            };
            let z: f64 = (0..i).map(|j| x[j] * k.values[j]).sum();
            let ss: f64 = (0..=i).map(|j| x[j] * -b[j]).sum();
            if ss.abs() < EPS {
                continue;
            }
            let s = ss.sqrt();
            if s > EPS {
                delta.push(k.values[i] - z);
                sigma.push(s);
            }
        }
        Ok((delta, sigma))
    }

    #[test]
    fn statistics_match_the_sequential_reference_bit_for_bit() {
        let mut s = 0x9e37_79b9_7f4a_7c15;
        let mut models = vec![line_data()];
        let scattered: Vec<Point> = (0..70)
            .map(|_| [uniform(&mut s) * 100.0, uniform(&mut s) * 100.0, 0.0])
            .collect();
        models.push(model_over(
            scattered.clone(),
            &mut s,
            Model::Spherical,
            vec![1.5, 40.0, 0.1],
        ));
        // A grid with a repeated point: equal distances and a zero distance.
        let mut grid: Vec<Point> = (0..7)
            .flat_map(|i| (0..7).map(move |j| [i as f64 * 5.0, j as f64 * 5.0, 0.0]))
            .collect();
        grid.push(grid[10]);
        models.push(model_over(
            grid,
            &mut s,
            Model::Exponential,
            vec![1.0, 30.0, 0.0],
        ));
        let mut geo = model_over(
            (0..50)
                .map(|_| {
                    [
                        uniform(&mut s) * 40.0 - 20.0,
                        uniform(&mut s) * 30.0 + 30.0,
                        0.0,
                    ]
                })
                .collect(),
            &mut s,
            Model::Gaussian,
            vec![1.0, 15.0, 0.05],
        );
        geo.geographic = true;
        models.push(geo);
        let cube: Vec<Point> = (0..50)
            .map(|_| {
                [
                    uniform(&mut s) * 10.0,
                    uniform(&mut s) * 10.0,
                    uniform(&mut s) * 10.0,
                ]
            })
            .collect();
        models.push(model_over(cube, &mut s, Model::Linear, vec![0.3, 0.02]));
        let mut pinv = model_over(
            scattered.into_iter().take(40).collect(),
            &mut s,
            Model::Spherical,
            vec![1.5, 40.0, 0.1],
        );
        pinv.inverse = Inverse::Pinv;
        models.push(pinv);
        let mut two = line_data();
        two.points.truncate(2);
        two.values.truncate(2);
        models.push(two);

        for (m, k) in models.iter().enumerate() {
            let (delta, sigma) = k.statistics().unwrap();
            assert!(
                !delta.is_empty() || k.points.len() <= 2,
                "model {m}: no residuals"
            );
            let (rd, rs) = reference_statistics(k).unwrap();
            assert_eq!(bits(&delta), bits(&rd), "model {m}: residuals");
            assert_eq!(bits(&sigma), bits(&rs), "model {m}: standard deviations");
        }
    }

    #[test]
    fn block_kriging_averages_the_right_hand_side_and_subtracts_the_block_variance() {
        let mut k = line_data();
        k.model = Model::Spherical;
        k.params = vec![1.0, 5.0, 0.1];
        // The model, its value at 0 being the nugget (used inside a block), and 0 on the
        // kriging matrix's diagonal.
        let model = |d: f64| {
            let r = (d / 5.0).min(1.0);
            1.0 * (1.5 * r - 0.5 * r.powi(3)) + 0.1
        };
        let gamma = |d: f64| if d == 0.0 { 0.0 } else { model(d) };
        // A block of two points 0.4 either side of the centre, weights 0.25 and 0.75.
        let block = [[-0.4, 0.0, 0.25], [0.4, 0.0, 0.75]];
        let targets = [[1.3, 0.0, 0.0], [2.6, 0.5, 0.0]];
        let (z, s) = k.execute_block(&targets, &[false, false], &block).unwrap();
        let a = Mat::from_fn(5, 5, |i, j| match (i < 4, j < 4) {
            (true, true) => -gamma(i.abs_diff(j) as f64),
            (false, false) => 0.0,
            _ => 1.0,
        });
        let block_variance =
            (0.25 * 0.25 + 0.75 * 0.75) * model(0.0) + 2.0 * 0.25 * 0.75 * model(0.8);
        for (t, p) in targets.iter().enumerate() {
            let b = Mat::from_fn(5, 1, |i, _| {
                if i == 4 {
                    1.0
                } else {
                    -block
                        .iter()
                        .map(|o| {
                            let d =
                                ((p[0] + o[0] - i as f64).powi(2) + (p[1] + o[1]).powi(2)).sqrt();
                            o[2] * model(d)
                        })
                        .sum::<f64>()
                }
            });
            let x = a.partial_piv_lu().solve(&b);
            let est = (0..4).map(|i| x[(i, 0)] * k.values[i]).sum::<f64>();
            let var = (0..5).map(|i| x[(i, 0)] * -b[(i, 0)]).sum::<f64>() - block_variance;
            assert!((z[t] - est).abs() < 1e-12, "target {t}: {} vs {est}", z[t]);
            assert!((s[t] - var).abs() < 1e-12, "target {t}: {} vs {var}", s[t]);
            // The estimate is the weighted mean of the point estimates (linearity).
            let points: Vec<Point> = block
                .iter()
                .map(|o| [p[0] + o[0], p[1] + o[1], 0.0])
                .collect();
            let (zp, _) = k.execute(&points, &[], &[false, false], None).unwrap();
            assert!((z[t] - (0.25 * zp[0] + 0.75 * zp[1])).abs() < 1e-12);
        }
        let mut uk = k.clone();
        uk.data_drift = vec![vec![0.0, 1.0, 2.0, 3.0]];
        assert!(uk.execute_block(&targets, &[false, false], &block).is_err());
    }

    #[test]
    fn gstools_models_are_kriging_variograms() {
        use crate::gstools::{cor, GsModel};
        let m = Model::Gstools(GsModel::TplStable);
        let params = [0.8, 30.0, 0.05, 0.5, 1.5];
        for d in [0.0, 3.0, 45.0] {
            let want = 0.8 * (1.0 - cor(GsModel::TplStable, d / 30.0, &[0.5, 1.5])) + 0.05;
            assert_eq!(value(m, &params, d), want);
        }
        assert_eq!(value(m, &params, 0.0), 0.05);
        assert!(fit(m, &[1.0, 2.0], &[0.1, 0.2], false).is_err());
    }

    #[test]
    fn simple_kriging_solves_the_covariance_system() {
        let mut k = line_data();
        k.model = Model::Exponential;
        k.params = vec![0.8, 6.0, 0.1];
        k.simple_mean = Some(2.0);
        let sill = 0.9;
        // gstools: var * cor off the diagonal, var + nugget on it (the nugget as measurement
        // error), the covariance with the nugget at distance 0 on the right (exact values).
        let cov = |d: f64| 0.8 * (-d / 2.0).exp();
        let targets = [[0.5, 0.0, 0.0], [2.0, 0.0, 0.0], [1e4, 3.0, 0.0]];
        let (z, s) = k.execute(&targets, &[], &[false; 3], None).unwrap();
        for (t, p) in targets.iter().enumerate() {
            let a = Mat::from_fn(4, 4, |i, j| {
                if i == j {
                    sill
                } else {
                    cov(i.abs_diff(j) as f64)
                }
            });
            let d = |i: usize| ((p[0] - i as f64).powi(2) + p[1].powi(2)).sqrt();
            let b = Mat::from_fn(4, 1, |i, _| if d(i) <= 1e-10 { sill } else { cov(d(i)) });
            let w = a.partial_piv_lu().solve(&b);
            let est = 2.0 + (0..4).map(|i| w[(i, 0)] * (k.values[i] - 2.0)).sum::<f64>();
            let var = (sill - (0..4).map(|i| w[(i, 0)] * b[(i, 0)]).sum::<f64>()).max(0.0);
            assert!((z[t] - est).abs() < 1e-12, "target {t}: {} vs {est}", z[t]);
            assert!((s[t] - var).abs() < 1e-12, "target {t}: {} vs {var}", s[t]);
        }
        assert!(
            (z[1] - 4.0).abs() < 1e-12 && s[1].abs() < 1e-12,
            "exact at a data point"
        );
        assert!(
            (z[2] - 2.0).abs() < 1e-12 && (s[2] - sill).abs() < 1e-12,
            "the mean far away"
        );
        let err = k.execute(&targets, &[], &[false; 3], Some(2)).unwrap_err();
        assert!(err.contains("simple kriging"), "{err}");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn oversized_systems_return_an_error_before_allocating() {
        let mut s = 0x1234_5678_9abc_def1;
        let points: Vec<Point> = (0..300_000)
            .map(|_| [uniform(&mut s) * 1e5, uniform(&mut s) * 1e5, 0.0])
            .collect();
        let k = model_over(points, &mut s, Model::Spherical, vec![1.0, 1e4, 0.0]);
        let targets = [[5e4, 5e4, 0.0]];
        let err = k.execute(&targets, &[], &[false], None).unwrap_err();
        assert!(
            err.contains("global kriging of 300000 points needs about")
                && err.contains("n_closest_points"),
            "{err}"
        );
        // A moving window stays small.
        assert!(k.execute(&targets, &[], &[false], Some(10)).is_ok());
        let err = k.statistics().unwrap_err();
        assert!(err.contains("statistics of 300000 points"), "{err}");
        let err = k.distances(&targets, &[false]).unwrap_err();
        assert!(err.contains("custom variogram"), "{err}");
    }
}
