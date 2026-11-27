//! Fitting a variogram model to an experimental variogram, after `skgstat.Variogram.fit`, and
//! the goodness-of-fit measures scikit-gstat reports.
//!
//! scikit-gstat calls `scipy.optimize.curve_fit` (Trust Region Reflective with bounds, or
//! Levenberg-Marquardt without) from the upper bounds. Here a bounded Levenberg-Marquardt with an
//! active set is run from scipy's start and from a few more fixed starts, and the lowest cost
//! wins: where scipy reaches the optimum the result is the same, and where it stops in a local
//! minimum (common with a nugget) the result is better.

use crate::models::{value, Model};

/// How the model is fitted (`fit_method`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Method {
    /// Bounded least squares (scipy's `trf`).
    Trf,
    /// Unbounded least squares (scipy's `lm`).
    Lm,
    /// scikit-gstat's maximum likelihood: `-sum(norm.logpdf(pred, y, 1) / sigma)` within the trf
    /// bounds, i.e. bounded least squares with each residual divided by `sqrt(sigma)`.
    Ml,
    /// Parameters given: `[range, sill, shape, nugget]` (shape ignored without one).
    Manual([f64; 4]),
}

/// Weights of the residuals (`fit_sigma`): each residual is divided by its sigma.
#[derive(Clone, Debug, PartialEq)]
pub enum Sigma {
    Unit,
    Linear,
    Exp,
    Sqrt,
    Sq,
    Values(Vec<f64>),
}

/// The fitting problem.
#[derive(Clone, Debug, PartialEq)]
pub struct FitOptions {
    pub model: Model,
    pub method: Method,
    pub sigma: Sigma,
    pub use_nugget: bool,
    /// scikit-gstat's `fit_bounds`: lower and upper bounds in the order of `cof`, replacing the
    /// default box (`trf`; `lm` is unbounded and `ml` bounded by `(0, p0)`).
    pub bounds: Option<(Vec<f64>, Vec<f64>)>,
    /// scikit-gstat's `fit_p0`: the initial guess in the order of `cof`, replacing scipy's start
    /// (the upper bounds); for `ml` it also sets the upper bounds and the start is half of it.
    pub p0: Option<Vec<f64>>,
}

/// A fitted model and its goodness of fit.
#[derive(Clone, Debug, PartialEq)]
pub struct Fitted {
    /// scikit-gstat's `cof`: `[r, c0, (s), (b)]`, the nugget only with `use_nugget` (always
    /// present for a manual fit).
    pub cof: Vec<f64>,
    /// `[effective_range, sill, (shape), nugget]`.
    pub parameters: Vec<f64>,
    /// The model at the lag edges.
    pub model_at_bins: Vec<f64>,
    pub rmse: f64,
    pub mse: f64,
    pub mae: f64,
    pub nrmse: f64,
    pub nrmse_r: f64,
    pub r: f64,
    pub nse: f64,
}

/// `cof` completed to the `[r, c0, (s), b]` form the model functions take.
pub fn full_cof(model: Model, cof: &[f64]) -> Vec<f64> {
    let base = if model.has_shape() { 3 } else { 2 };
    let mut out = cof.to_vec();
    if out.len() == base {
        out.push(0.0);
    }
    out
}

fn sigma_values(sigma: &Sigma, bins: &[f64]) -> Result<Vec<f64>, String> {
    let max = bins.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let rel = |b: f64| b / max;
    Ok(match sigma {
        Sigma::Unit => vec![1.0; bins.len()],
        Sigma::Linear => bins.iter().map(|&b| rel(b)).collect(),
        Sigma::Exp => bins.iter().map(|&b| 1.0 / (1.0 / rel(b)).exp()).collect(),
        Sigma::Sqrt => bins.iter().map(|&b| rel(b).sqrt()).collect(),
        Sigma::Sq => bins.iter().map(|&b| rel(b).powi(2)).collect(),
        Sigma::Values(v) if v.len() == bins.len() => v.clone(),
        Sigma::Values(_) => return Err("fit_sigma must have one value per lag".to_string()),
    })
}

struct Problem<'a> {
    model: Model,
    x: &'a [f64],
    y: &'a [f64],
    sigma: &'a [f64],
    use_nugget: bool,
    lb: Vec<f64>,
    ub: Vec<f64>,
    /// Typical size of each parameter (its upper bound), for the trust region.
    scale: Vec<f64>,
    bounded: bool,
}

impl Problem<'_> {
    fn residuals(&self, p: &[f64]) -> Vec<f64> {
        let cof = if self.use_nugget {
            p.to_vec()
        } else {
            full_cof(self.model, p)
        };
        self.x
            .iter()
            .zip(self.y)
            .zip(self.sigma)
            .map(|((&h, &y), &s)| (value(self.model, h, &cof) - y) / s)
            .collect()
    }

    fn cost(&self, p: &[f64]) -> f64 {
        let r = self.residuals(p);
        if r.iter().any(|v| !v.is_finite()) {
            return f64::INFINITY;
        }
        0.5 * r.iter().map(|v| v * v).sum::<f64>()
    }

    /// The best sill and nugget for a fixed range (and shape): the model is linear in them, so
    /// they solve a small weighted least-squares problem, bounded for the bounded fit.
    fn linear_part(&self, range: f64, shape: f64, max_y: f64) -> Vec<f64> {
        // Basis: model with sill 1 and nugget 0 (g), and the constant 1 (nugget).
        let unit: Vec<f64> = if self.model.has_shape() {
            vec![range, 1.0, shape, 0.0]
        } else {
            vec![range, 1.0, 0.0]
        };
        let g: Vec<f64> = self
            .x
            .iter()
            .map(|&h| value(self.model, h, &unit))
            .collect();
        let w: Vec<f64> = self.sigma.iter().map(|s| 1.0 / (s * s)).collect();
        let sum = |f: &dyn Fn(usize) -> f64| (0..self.x.len()).map(f).sum::<f64>();
        let (sgg, sgy) = (
            sum(&|i| w[i] * g[i] * g[i]),
            sum(&|i| w[i] * g[i] * self.y[i]),
        );
        let cap = |v: f64, hi: f64| if self.bounded { v.clamp(0.0, hi) } else { v };
        let sill_only = cap(if sgg > 0.0 { sgy / sgg } else { 0.0 }, max_y);
        let params = |sill: f64, nugget: Option<f64>| {
            let mut p = vec![range, sill];
            if self.model.has_shape() {
                p.push(shape);
            }
            if let Some(n) = nugget {
                p.push(n);
            }
            p
        };
        if !self.use_nugget {
            return params(sill_only, None);
        }
        let (sg, s1, sy) = (
            sum(&|i| w[i] * g[i]),
            sum(&|i| w[i]),
            sum(&|i| w[i] * self.y[i]),
        );
        let det = sgg * s1 - sg * sg;
        let nugget_hi = 0.99 * max_y;
        let mut candidates = vec![params(sill_only, Some(0.0))];
        if det.abs() > 1e-300 {
            let sill = (sgy * s1 - sg * sy) / det;
            let nugget = (sgg * sy - sg * sgy) / det;
            candidates.push(params(cap(sill, max_y), Some(cap(nugget, nugget_hi))));
        }
        // Nugget alone (sill 0) and the nugget at its upper bound.
        candidates.push(params(0.0, Some(cap(sy / s1, nugget_hi))));
        let nugget = nugget_hi;
        let sill = cap(
            if sgg > 0.0 {
                (sgy - nugget * sg) / sgg
            } else {
                0.0
            },
            max_y,
        );
        candidates.push(params(sill, Some(nugget)));
        candidates
            .into_iter()
            .min_by(|a, b| self.cost(a).total_cmp(&self.cost(b)))
            .expect("candidates")
    }

    /// Starting points from a scan of the range (and the shape): the `keep` best grid points,
    /// each with its optimal sill and nugget.
    fn grid_starts(&self, max_x: f64, max_y: f64, keep: usize) -> Vec<Vec<f64>> {
        let shapes: Vec<f64> = match self.model {
            Model::Stable => vec![0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
            Model::Matern => vec![
                0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0,
            ],
            // GSTools models: a geometric scan within the shape's bounds.
            Model::Gstools(_) if self.model.has_shape() => {
                let lo = self.model.shape_lower().max(0.05);
                let hi = self.model.shape_bound().min(20.0);
                (0..12)
                    .map(|i| lo * (hi / lo).powf(i as f64 / 11.0))
                    .collect()
            }
            _ => vec![1.0],
        };
        let mut points: Vec<(Vec<f64>, f64)> = Vec::new();
        for k in 1..=80 {
            let range = max_x * k as f64 / 80.0;
            for &shape in &shapes {
                let p = self.linear_part(range, shape, max_y);
                let c = self.cost(&p);
                if c.is_finite() {
                    points.push((p, c));
                }
            }
        }
        points.sort_by(|a, b| a.1.total_cmp(&b.1));
        points.into_iter().take(keep).map(|(p, _)| p).collect()
    }

    /// Bounded (or unbounded) Levenberg-Marquardt from `p` (see [`crate::lsq::minimise`]).
    fn minimise(&self, p: Vec<f64>) -> (Vec<f64>, f64) {
        let k = p.len();
        let (lb, ub) = if self.bounded {
            (self.lb.clone(), self.ub.clone())
        } else {
            (vec![f64::NEG_INFINITY; k], vec![f64::INFINITY; k])
        };
        let residuals = |q: &[f64]| self.residuals(q);
        let (p, cost) = crate::lsq::minimise(&residuals, p, &lb, &ub, &self.scale);
        (p, 0.5 * cost)
    }
}

fn nanmean(v: &[f64]) -> f64 {
    let kept: Vec<f64> = v.iter().copied().filter(|x| !x.is_nan()).collect();
    if kept.is_empty() {
        f64::NAN
    } else {
        kept.iter().sum::<f64>() / kept.len() as f64
    }
}

fn nansum(v: impl Iterator<Item = f64>) -> f64 {
    v.filter(|x| !x.is_nan()).sum()
}

/// Fits `opts.model` to the experimental variogram (lag upper edges `bins`).
pub fn fit(bins: &[f64], experimental: &[f64], opts: &FitOptions) -> Result<Fitted, String> {
    let model = opts.model;
    let cof = match opts.method {
        Method::Manual([range, sill, shape, nugget]) => {
            let nugget = if nugget.is_nan() { 0.0 } else { nugget };
            if model.has_shape() {
                vec![range, sill, shape, nugget]
            } else {
                vec![range, sill, nugget]
            }
        }
        method => {
            let mut sigma_all = sigma_values(&opts.sigma, bins)?;
            if method == Method::Ml {
                sigma_all.iter_mut().for_each(|s| *s = s.sqrt());
            }
            let bounded = matches!(method, Method::Trf | Method::Ml);
            let keep: Vec<usize> = (0..bins.len())
                .filter(|&i| !experimental[i].is_nan())
                .collect();
            if keep.len() < 2 {
                return Err("a fit needs at least 2 non-empty lags".to_string());
            }
            let x: Vec<f64> = keep.iter().map(|&i| bins[i]).collect();
            let y: Vec<f64> = keep.iter().map(|&i| experimental[i]).collect();
            let sigma: Vec<f64> = keep.iter().map(|&i| sigma_all[i]).collect();
            // scikit-gstat bounds the range by the last lag edge, empty lags included.
            let max_x = bins.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let max_y = y.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            // The bounds [0, max] are empty below 0 (a classical cross-variogram of opposite
            // variables); scipy refuses them too.
            if bounded && max_y < 0.0 {
                return Err(
                    "a bounded fit (fit_method: :trf or :ml) needs a positive semivariance; \
                     use fit_method: :lm or nil"
                        .to_string(),
                );
            }
            let mut ub = vec![max_x, max_y];
            if model.has_shape() {
                ub.push(model.shape_bound());
            }
            if opts.use_nugget {
                ub.push(0.99 * max_y);
            }
            let k = ub.len();
            let mut lb: Vec<f64> = (0..k)
                .map(|i| {
                    if i == 2 && model.has_shape() {
                        model.shape_lower()
                    } else {
                        0.0
                    }
                })
                .collect();
            if let Some((l, u)) = &opts.bounds {
                if l.len() != k || u.len() != k {
                    return Err(format!("fit_bounds need {k} lower and {k} upper values"));
                }
            }
            if opts.p0.as_ref().is_some_and(|p| p.len() != k) {
                return Err(format!("fit_p0 needs {k} values"));
            }
            match (method, &opts.bounds, &opts.p0) {
                // scikit-gstat's maximum likelihood is bounded by (0, p0), whatever fit_bounds.
                (Method::Ml, _, Some(p0)) => {
                    ub = p0.clone();
                    lb = vec![0.0; k];
                }
                (Method::Ml, _, None) => {}
                (_, Some((l, u)), _) => {
                    lb = l.clone();
                    ub = u.clone();
                }
                _ => {}
            }
            if bounded && lb.iter().zip(&ub).any(|(l, u)| l > u) {
                return Err("every lower bound must not exceed its upper bound".to_string());
            }
            let problem = Problem {
                model,
                x: &x,
                y: &y,
                sigma: &sigma,
                use_nugget: opts.use_nugget,
                lb: lb.clone(),
                scale: ub.iter().map(|u| u.abs().max(1e-12)).collect(),
                ub: ub.clone(),
                bounded,
            };
            // scipy's start: the upper bounds (or the user's p0; half of it for ml), moved
            // strictly inside for the bounded fit.
            let start: Vec<f64> = match (method, &opts.p0) {
                (Method::Ml, Some(p0)) => p0.iter().map(|v| v * 0.5).collect(),
                (_, Some(p0)) => p0.clone(),
                _ => ub.clone(),
            };
            let scipy: Vec<f64> = if bounded {
                start
                    .iter()
                    .zip(lb.iter().zip(&ub))
                    .map(|(&v, (&l, &u))| {
                        if v >= u {
                            u - 1e-10 * u.abs().max(1.0)
                        } else if v <= l {
                            l + 1e-10 * l.abs().max(1.0)
                        } else {
                            v
                        }
                    })
                    .collect()
            } else {
                start
            };
            let user = opts.bounds.is_some() || opts.p0.is_some();
            let mut starts = vec![scipy];
            starts.extend(problem.grid_starts(max_x, max_y, 3).into_iter().map(|p| {
                if bounded && user {
                    p.iter()
                        .zip(lb.iter().zip(&ub))
                        .map(|(&v, (&l, &u))| v.clamp(l, u))
                        .collect()
                } else {
                    p
                }
            }));
            starts
                .into_iter()
                .map(|p| problem.minimise(p))
                .filter(|(_, c)| c.is_finite())
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(p, _)| p)
                .ok_or("the fit did not converge")?
        }
    };
    Ok(evaluate(model, cof, bins, experimental, opts.use_nugget))
}

fn evaluate(
    model: Model,
    cof: Vec<f64>,
    bins: &[f64],
    experimental: &[f64],
    use_nugget: bool,
) -> Fitted {
    let full = full_cof(model, &cof);
    let model_at_bins: Vec<f64> = bins.iter().map(|&h| value(model, h, &full)).collect();
    let mut parameters = vec![full[0], full[1]];
    if model.has_shape() {
        parameters.push(full[2]);
    }
    let manual_nugget = cof.len() == full.len() && !use_nugget && full[full.len() - 1] != 0.0;
    parameters.push(if use_nugget || manual_nugget {
        full[full.len() - 1]
    } else {
        0.0
    });
    with_diagnostics(cof, parameters, model_at_bins, experimental)
}

/// The goodness of fit of `model_at_bins` against the experimental values, as scikit-gstat
/// computes it (NaN lags left out of the means and sums).
fn with_diagnostics(
    cof: Vec<f64>,
    parameters: Vec<f64>,
    model_at_bins: Vec<f64>,
    experimental: &[f64],
) -> Fitted {
    let res: Vec<f64> = model_at_bins
        .iter()
        .zip(experimental)
        .map(|(m, e)| m - e)
        .collect();
    let sq: Vec<f64> = res.iter().map(|v| v * v).collect();
    let rmse = nanmean(&sq).sqrt();
    let mse = sq.iter().sum::<f64>() / sq.len() as f64;
    let mae = res.iter().map(|v| v.abs()).sum::<f64>() / res.len() as f64;
    let mean_exp = nanmean(experimental);
    let max_exp = experimental
        .iter()
        .copied()
        .filter(|v| !v.is_nan())
        .fold(f64::NEG_INFINITY, f64::max);
    let mean_mod = nanmean(&model_at_bins);
    let pairs = || experimental.iter().zip(&model_at_bins);
    let term1 = nansum(pairs().map(|(x, y)| (x - mean_exp) * (y - mean_mod)));
    let t2x = nansum(experimental.iter().map(|x| (x - mean_exp).powi(2)));
    let t2y = nansum(model_at_bins.iter().map(|y| (y - mean_mod).powi(2)));
    let ns_num = nansum(pairs().map(|(x, y)| (x - y).powi(2)));
    Fitted {
        cof,
        parameters,
        rmse,
        mse,
        mae,
        nrmse: rmse / mean_exp,
        nrmse_r: rmse / (max_exp - mean_exp),
        r: term1 / (t2x * t2y).sqrt(),
        nse: 1.0 - ns_num / t2x,
        model_at_bins,
    }
}

/// The number of coefficients of a summed model without its nugget: each component's range,
/// sill and (for `stable` and `matern`) shape.
pub fn sum_size(models: &[Model]) -> usize {
    models
        .iter()
        .map(|m| if m.has_shape() { 3 } else { 2 })
        .sum()
}

/// A summed model (scikit-gstat's `"spherical+gaussian"`) at `h`: the components with their
/// `(range, sill[, shape])` in order, plus the trailing nugget when `cof` has one.
pub fn sum_value(models: &[Model], h: f64, cof: &[f64]) -> f64 {
    let mut at = 0;
    let mut total = 0.0;
    for &m in models {
        let k = if m.has_shape() { 3 } else { 2 };
        let mut part = cof[at..at + k].to_vec();
        part.push(0.0);
        total += value(m, h, &part);
        at += k;
    }
    total + cof.get(at).copied().unwrap_or(0.0)
}

/// Shapes tried for a component when scanning starting points.
fn scan_shapes(model: Model) -> Vec<f64> {
    match model {
        Model::Stable => vec![0.5, 1.0, 1.5, 2.0],
        Model::Matern => vec![0.5, 1.5, 5.0],
        _ => vec![f64::NAN],
    }
}

/// Fits a summed model within scikit-gstat's bounds (each component's
/// `[max lag, max semivariance, (shape bound)]`, the nugget below `0.99 * max semivariance`):
/// a scan of the ranges and shapes, with the sills and nugget solved by linear least squares at
/// each point, refined by a bounded (or unbounded, for `:lm`) Levenberg-Marquardt from the best
/// points and from scipy's start.
pub fn fit_sum(
    models: &[Model],
    bins: &[f64],
    experimental: &[f64],
    method: Method,
    sigma: &Sigma,
    use_nugget: bool,
) -> Result<Fitted, String> {
    if models.len() < 2 {
        return Err("a summed model needs at least 2 components".to_string());
    }
    if matches!(method, Method::Manual(_)) {
        return Err("a summed model cannot be fitted manually".to_string());
    }
    let mut sigma_all = sigma_values(sigma, bins)?;
    if method == Method::Ml {
        sigma_all.iter_mut().for_each(|s| *s = s.sqrt());
    }
    let bounded = matches!(method, Method::Trf | Method::Ml);
    let keep: Vec<usize> = (0..bins.len())
        .filter(|&i| !experimental[i].is_nan())
        .collect();
    if keep.len() < 2 {
        return Err("a fit needs at least 2 non-empty lags".to_string());
    }
    let x: Vec<f64> = keep.iter().map(|&i| bins[i]).collect();
    let y: Vec<f64> = keep.iter().map(|&i| experimental[i]).collect();
    let w: Vec<f64> = keep.iter().map(|&i| sigma_all[i]).collect();
    let max_x = bins.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let max_y = y.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if bounded && max_y < 0.0 {
        return Err(
            "a bounded fit (fit_method: :trf or :ml) needs a positive semivariance; \
             use fit_method: :lm or nil"
                .to_string(),
        );
    }
    let mut ub = Vec::new();
    for &m in models {
        ub.extend([max_x, max_y]);
        if m.has_shape() {
            ub.push(m.shape_bound());
        }
    }
    if use_nugget {
        ub.push(0.99 * max_y);
    }
    let k = ub.len();
    let residuals = |p: &[f64]| -> Vec<f64> {
        x.iter()
            .zip(&y)
            .zip(&w)
            .map(|((&h, &v), &s)| (sum_value(models, h, p) - v) / s)
            .collect()
    };
    let cost = |p: &[f64]| {
        let r = residuals(p);
        if r.iter().all(|v| v.is_finite()) {
            r.iter().map(|v| v * v).sum::<f64>()
        } else {
            f64::INFINITY
        }
    };
    // The sills (and nugget) for fixed ranges and shapes: weighted linear least squares. For the
    // bounded fit each subset of them may sit at 0 (a component scipy switches off, or two equal
    // components whose system is singular): the best feasible subset, upper bounds clipped.
    let linear = |ranges: &[f64], shapes: &[f64]| -> Option<Vec<f64>> {
        let basis: Vec<Vec<f64>> = models
            .iter()
            .enumerate()
            .map(|(c, &m)| {
                let mut unit = vec![ranges[c], 1.0];
                if m.has_shape() {
                    unit.push(shapes[c]);
                }
                unit.push(0.0);
                x.iter().map(|&h| value(m, h, &unit)).collect()
            })
            .chain(use_nugget.then(|| vec![1.0; x.len()]))
            .collect();
        let n = basis.len();
        let weight: Vec<f64> = w.iter().map(|s| 1.0 / (s * s)).collect();
        let dot = |a: &[f64], b: &[f64]| -> f64 {
            a.iter()
                .zip(b)
                .zip(&weight)
                .map(|((u, v), q)| u * v * q)
                .sum()
        };
        let upper = |i: usize| {
            if i < models.len() {
                max_y
            } else {
                0.99 * max_y
            }
        };
        let assemble = |coef: &[f64]| {
            let mut p = Vec::with_capacity(k);
            for (c, &m) in models.iter().enumerate() {
                p.extend([ranges[c], coef[c]]);
                if m.has_shape() {
                    p.push(shapes[c]);
                }
            }
            if use_nugget {
                p.push(coef[n - 1]);
            }
            p
        };
        let masks: Vec<u32> = if bounded {
            (0..1 << n).collect()
        } else {
            vec![0]
        };
        masks
            .into_iter()
            .filter_map(|zero| {
                let free: Vec<usize> = (0..n).filter(|i| zero & (1 << i) == 0).collect();
                let mut coef = vec![0.0; n];
                if !free.is_empty() {
                    let a: Vec<Vec<f64>> = free
                        .iter()
                        .map(|&i| free.iter().map(|&j| dot(&basis[i], &basis[j])).collect())
                        .collect();
                    let b: Vec<f64> = free.iter().map(|&i| dot(&basis[i], &y)).collect();
                    let sol = crate::lsq::solve_small(a, b)?;
                    for (&i, v) in free.iter().zip(sol) {
                        if bounded && v < 0.0 {
                            return None;
                        }
                        coef[i] = if bounded { v.min(upper(i)) } else { v };
                    }
                }
                let p = assemble(&coef);
                let c = cost(&p);
                c.is_finite().then_some((p, c))
            })
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(p, _)| p)
    };
    // The scan: ranges on a grid per component (coarser with more components) and each
    // component's scan shapes.
    let steps = match models.len() {
        2 => 24,
        3 => 10,
        _ => 5,
    };
    let axes: Vec<Vec<(f64, f64)>> = models
        .iter()
        .map(|&m| {
            (1..=steps)
                .flat_map(|j| {
                    let range = max_x * j as f64 / steps as f64;
                    scan_shapes(m).into_iter().map(move |shape| (range, shape))
                })
                .collect()
        })
        .collect();
    let mut scanned: Vec<(Vec<f64>, f64)> = Vec::new();
    let mut index = vec![0usize; models.len()];
    loop {
        let point: Vec<(f64, f64)> = index.iter().zip(&axes).map(|(&i, a)| a[i]).collect();
        let ranges: Vec<f64> = point.iter().map(|p| p.0).collect();
        let shapes: Vec<f64> = point.iter().map(|p| p.1).collect();
        if let Some(p) = linear(&ranges, &shapes) {
            let c = cost(&p);
            if c.is_finite() {
                scanned.push((p, c));
            }
        }
        // Next index of the cartesian product.
        let mut d = 0;
        while d < index.len() {
            index[d] += 1;
            if index[d] < axes[d].len() {
                break;
            }
            index[d] = 0;
            d += 1;
        }
        if d == index.len() {
            break;
        }
    }
    scanned.sort_by(|a, b| a.1.total_cmp(&b.1));
    let scipy: Vec<f64> = if bounded {
        ub.iter().map(|&u| u - 1e-10 * u.abs().max(1.0)).collect()
    } else {
        ub.clone()
    };
    let (lb, hi) = if bounded {
        (vec![0.0; k], ub.clone())
    } else {
        (vec![f64::NEG_INFINITY; k], vec![f64::INFINITY; k])
    };
    let scale: Vec<f64> = ub.iter().map(|u| u.abs().max(1e-12)).collect();
    let best = std::iter::once(scipy)
        .chain(scanned.into_iter().take(10).map(|(p, _)| p))
        .map(|p| crate::lsq::minimise(&residuals, p, &lb, &hi, &scale))
        .filter(|(_, c)| c.is_finite())
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(p, _)| p)
        .ok_or("the fit did not converge")?;
    // A second pass from the best point (the trust region restarts at full size).
    let polished = crate::lsq::minimise(&residuals, best.clone(), &lb, &hi, &scale);
    let cof = if polished.1 <= cost(&best) {
        polished.0
    } else {
        best
    };
    let model_at_bins: Vec<f64> = bins.iter().map(|&h| sum_value(models, h, &cof)).collect();
    let mut parameters = cof.clone();
    if !use_nugget {
        parameters.push(0.0);
    }
    Ok(with_diagnostics(
        cof,
        parameters,
        model_at_bins,
        experimental,
    ))
}

/// sklearn's increasing isotonic regression with unit weights (its in-place pool adjacent
/// violators: a block pools while the next value is not strictly larger, then backtracks).
pub fn isotonic(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut y = values.to_vec();
    let mut w = vec![1.0; n];
    // target[i] = j and target[j] = i for an active block [i..=j].
    let mut target: Vec<usize> = (0..n).collect();
    let mut i = 0;
    while i < n {
        let mut k = target[i] + 1;
        if k == n {
            break;
        }
        if y[i] < y[k] {
            i = k;
            continue;
        }
        let mut sum_wy = w[i] * y[i];
        let mut sum_w = w[i];
        loop {
            let prev_y = y[k];
            sum_wy += w[k] * y[k];
            sum_w += w[k];
            k = target[k] + 1;
            if k == n || prev_y < y[k] {
                y[i] = sum_wy / sum_w;
                w[i] = sum_w;
                target[i] = k - 1;
                target[k - 1] = i;
                if i > 0 {
                    i = target[i - 1];
                }
                break;
            }
        }
    }
    let mut i = 0;
    while i < n {
        let k = target[i] + 1;
        let v = y[i];
        y[i + 1..k].iter_mut().for_each(|x| *x = v);
        i = k;
    }
    y
}

/// The harmonized model at `h`: the isotonic values `y` at the lags `x` interpolated linearly
/// (scipy's `interp1d`), NaN outside `[x_first, x_last]`.
pub fn harmonize_value(x: &[f64], y: &[f64], h: f64) -> f64 {
    let (Some(&first), Some(&last)) = (x.first(), x.last()) else {
        return f64::NAN;
    };
    if h.is_nan() || h < first || h > last {
        return f64::NAN;
    }
    if x.len() == 1 {
        return y[0];
    }
    // interp1d: the interval found by a left search, clipped to [1, n - 1].
    let hi = x.partition_point(|&v| v < h).clamp(1, x.len() - 1);
    let lo = hi - 1;
    let slope = (y[hi] - y[lo]) / (x[hi] - x[lo]);
    slope * (h - x[lo]) + y[lo]
}

/// scikit-gstat's harmonized variogram (`model="harmonize"`): the isotonic regression of the
/// non-empty lags, interpolated. `cof` is `[range, sill, nugget]` from
/// `x = linspace(0, last kept lag, 100)`: `sill = 0.95 * max(model(x))`, `range` the first `x`
/// reaching it (scikit-gstat stores its index instead) and, with `use_nugget`, `nugget = model(0)`
/// (NaN unless a lag starts at 0).
pub fn harmonize(bins: &[f64], experimental: &[f64], use_nugget: bool) -> Result<Fitted, String> {
    let keep: Vec<usize> = (0..bins.len())
        .filter(|&i| !experimental[i].is_nan())
        .collect();
    if keep.is_empty() {
        return Err("a harmonized variogram needs a non-empty lag".to_string());
    }
    let x: Vec<f64> = keep.iter().map(|&i| bins[i]).collect();
    let y = isotonic(&keep.iter().map(|&i| experimental[i]).collect::<Vec<_>>());
    let model_at_bins: Vec<f64> = bins.iter().map(|&h| harmonize_value(&x, &y, h)).collect();
    let grid = crate::numeric::linspace(0.0, x[x.len() - 1], 100);
    let values: Vec<f64> = grid.iter().map(|&h| harmonize_value(&x, &y, h)).collect();
    let sill = 0.95
        * values
            .iter()
            .copied()
            .filter(|v| !v.is_nan())
            .fold(f64::NEG_INFINITY, f64::max);
    let r = values
        .iter()
        .position(|&v| v >= sill)
        .ok_or("the harmonized model never reaches its sill")?;
    let nugget = if use_nugget { values[0] } else { 0.0 };
    let cof = vec![grid[r], sill, nugget];
    Ok(with_diagnostics(
        cof.clone(),
        cof,
        model_at_bins,
        experimental,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts(model: Model, method: Method, use_nugget: bool) -> FitOptions {
        FitOptions {
            model,
            method,
            sigma: Sigma::Unit,
            use_nugget,
            bounds: None,
            p0: None,
        }
    }

    #[test]
    fn user_bounds_and_start_control_the_fit() {
        let bins: Vec<f64> = (1..=12).map(|i| i as f64 * 10.0).collect();
        let truth = [80.0, 2.0, 0.3];
        let experimental: Vec<f64> = bins
            .iter()
            .map(|&h| value(Model::Spherical, h, &truth))
            .collect();
        // A range bound below the truth binds; the sill and nugget adapt within theirs.
        let bounded = FitOptions {
            bounds: Some((vec![0.0, 0.0, 0.5], vec![60.0, 5.0, 1.0])),
            ..opts(Model::Spherical, Method::Trf, true)
        };
        let f = fit(&bins, &experimental, &bounded).unwrap();
        assert_eq!(f.cof[0], 60.0);
        assert!(f.cof[2] >= 0.5 && f.cof[2] <= 1.0, "{:?}", f.cof);
        // A start near the truth with loose bounds finds it.
        let started = FitOptions {
            bounds: Some((vec![0.0, 0.0, 0.0], vec![1000.0, 50.0, 10.0])),
            p0: Some(vec![70.0, 1.5, 0.2]),
            ..opts(Model::Spherical, Method::Trf, true)
        };
        let f = fit(&bins, &experimental, &started).unwrap();
        for (a, b) in f.cof.iter().zip(truth) {
            assert!((a - b).abs() < 1e-6 * b.max(1.0), "{:?}", f.cof);
        }
        // Maximum likelihood: p0 is also the upper bound.
        let ml = FitOptions {
            p0: Some(vec![50.0, 3.0, 1.0]),
            ..opts(Model::Spherical, Method::Ml, true)
        };
        assert_eq!(fit(&bins, &experimental, &ml).unwrap().cof[0], 50.0);
        // Sizes must match the model's parameters.
        let wrong = FitOptions {
            p0: Some(vec![1.0, 2.0]),
            ..opts(Model::Spherical, Method::Trf, true)
        };
        assert!(fit(&bins, &experimental, &wrong).is_err());
    }

    #[test]
    fn summed_models_add_their_components() {
        let models = [Model::Spherical, Model::Stable];
        let cof = [300.0, 1.0, 800.0, 2.0, 1.5, 0.25];
        let h = 100.0;
        let parts = value(Model::Spherical, h, &[300.0, 1.0, 0.0])
            + value(Model::Stable, h, &[800.0, 2.0, 1.5, 0.0]);
        assert!((sum_value(&models, h, &cof) - (parts + 0.25)).abs() < 1e-15);
        // Without the trailing nugget it is 0.
        assert!((sum_value(&models, h, &cof[..5]) - parts).abs() < 1e-15);
        assert_eq!(sum_size(&models), 5);
    }

    #[test]
    fn a_summed_fit_recovers_its_components() {
        let models = [Model::Spherical, Model::Exponential];
        let truth = [5.0, 1.0, 15.0, 2.0, 0.3];
        let bins: Vec<f64> = (1..=20).map(f64::from).collect();
        let exp: Vec<f64> = bins
            .iter()
            .map(|&h| sum_value(&models, h, &truth))
            .collect();
        let f = fit_sum(&models, &bins, &exp, Method::Trf, &Sigma::Unit, true).unwrap();
        assert!(f.rmse < 1e-6, "{:?} rmse {}", f.cof, f.rmse);
        assert_eq!((f.cof.len(), f.parameters.len()), (5, 5));
        let plain = fit_sum(&models, &bins, &exp, Method::Trf, &Sigma::Unit, false).unwrap();
        // parameters always end with the nugget, 0 without one.
        assert_eq!((plain.cof.len(), plain.parameters.len()), (4, 5));
        assert_eq!(plain.parameters[4], 0.0);
        let manual = Method::Manual([1.0, 1.0, 1.0, 0.0]);
        assert!(fit_sum(&models, &bins, &exp, manual, &Sigma::Unit, false).is_err());
    }

    #[test]
    fn isotonic_regression_pools_adjacent_violators() {
        assert_eq!(
            isotonic(&[1.0, 3.0, 2.0, 4.0, 3.0, 5.0]),
            vec![1.0, 2.5, 2.5, 3.5, 3.5, 5.0]
        );
        // A long decreasing run pools into its mean, then backtracks into the earlier block.
        assert_eq!(isotonic(&[3.0, 1.0, 2.0, 0.0]), vec![1.5, 1.5, 1.5, 1.5]);
        assert_eq!(isotonic(&[1.0, 2.0, 3.0]), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn harmonize_interpolates_the_isotonic_fit() {
        // Lag 3 is empty: the model interpolates across it and is NaN beyond the kept lags.
        let bins = [1.0, 2.0, 3.0, 4.0, 5.0];
        let exp = [2.0, 1.0, f64::NAN, 4.0, f64::NAN];
        let f = harmonize(&bins, &exp, false).unwrap();
        assert_eq!(f.model_at_bins[..2], [1.5, 1.5]);
        assert!((f.model_at_bins[2] - 2.75).abs() < 1e-12);
        assert_eq!(f.model_at_bins[3], 4.0);
        assert!(f.model_at_bins[4].is_nan());
        // The model passes 0.95 * 4 = 3.8 at h = 3.84; the first point of linspace(0, 4, 100)
        // beyond is index 96 (h = 3.879; index 95, h = 3.838, gives 3.798).
        assert!((f.cof[0] - 4.0 * 96.0 / 99.0).abs() < 1e-12, "{:?}", f.cof);
        assert_eq!(f.cof[1..], [3.8, 0.0]);
        assert!(harmonize(&bins, &exp, true).unwrap().cof[2].is_nan());
        assert_eq!(harmonize_value(&[1.0, 2.0], &[1.0, 3.0], 1.5), 2.0);
        assert!(harmonize_value(&[1.0, 2.0], &[1.0, 3.0], 0.5).is_nan());
    }

    #[test]
    fn ml_is_the_bounded_fit_with_square_root_sigma() {
        // scikit-gstat's ml minimises sum((pred - y)^2 / sigma) within the trf bounds.
        let bins = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let exp = [0.9, 1.7, 2.6, 3.0, 3.3, 3.2];
        let mut ml = opts(Model::Exponential, Method::Ml, true);
        ml.sigma = Sigma::Linear;
        let mut trf = opts(Model::Exponential, Method::Trf, true);
        trf.sigma = Sigma::Values(bins.iter().map(|b| (b / 6.0_f64).sqrt()).collect());
        let (a, b) = (
            fit(&bins, &exp, &ml).unwrap(),
            fit(&bins, &exp, &trf).unwrap(),
        );
        for (x, y) in a.cof.iter().zip(&b.cof) {
            assert!(
                (x - y).abs() <= 1e-9 * y.abs().max(1.0),
                "{:?} vs {:?}",
                a.cof,
                b.cof
            );
        }
        let negative = [-0.5, -0.7, -0.9];
        let ml = opts(Model::Spherical, Method::Ml, false);
        assert!(fit(&bins[..3], &negative, &ml).is_err());
    }

    #[test]
    fn a_bounded_fit_refuses_negative_semivariances() {
        // A classical cross-variogram of opposite variables: the sill bounds [0, max] are empty,
        // which scipy refuses too. An error, not a panic.
        let bins = [1.0, 2.0, 3.0];
        let exp = [-0.5, -0.7, -0.9];
        let err = fit(&bins, &exp, &opts(Model::Spherical, Method::Trf, false)).unwrap_err();
        assert!(err.contains("positive"), "{err}");
    }

    #[test]
    fn the_range_may_reach_an_empty_last_lag() {
        // Still rising at the last non-empty lag: the best range lies beyond it, up to the last
        // edge, which bounds it as in scikit-gstat.
        let bins = [1.0, 2.0, 3.0, 4.0];
        let exp = [1.0, 2.0, 3.0, f64::NAN];
        let f = fit(&bins, &exp, &opts(Model::Spherical, Method::Trf, false)).unwrap();
        assert!(f.cof[0] > 3.0 && f.cof[0] <= 4.0, "range {}", f.cof[0]);
    }

    #[test]
    fn recovers_an_exact_spherical_variogram() {
        let truth = [6.0, 2.0, 0.5];
        let bins: Vec<f64> = (1..=12).map(|i| i as f64).collect();
        let exp: Vec<f64> = bins
            .iter()
            .map(|&h| value(Model::Spherical, h, &truth))
            .collect();
        let f = fit(&bins, &exp, &opts(Model::Spherical, Method::Trf, true)).unwrap();
        for (a, b) in f.cof.iter().zip(truth) {
            assert!((a - b).abs() < 1e-6, "{:?}", f.cof);
        }
        assert!(f.rmse < 1e-8 && (f.nse - 1.0).abs() < 1e-12);
    }

    #[test]
    fn a_nugget_fit_is_never_worse_than_the_fit_without_it() {
        let bins: Vec<f64> = (1..=10).map(|i| i as f64 * 10.0).collect();
        let exp = [3.0, 5.5, 6.0, 8.1, 7.9, 8.5, 9.0, 8.8, 9.2, 9.1];
        let plain = fit(&bins, &exp, &opts(Model::Cubic, Method::Trf, false)).unwrap();
        let nugget = fit(&bins, &exp, &opts(Model::Cubic, Method::Trf, true)).unwrap();
        assert!(nugget.rmse <= plain.rmse * (1.0 + 1e-9));
    }

    #[test]
    fn manual_parameters_are_kept_and_the_nugget_reported() {
        let bins = [1.0, 2.0, 3.0];
        let exp = [1.0, 2.0, 2.5];
        let f = fit(
            &bins,
            &exp,
            &opts(
                Model::Stable,
                Method::Manual([2.5, 2.0, 1.5, f64::NAN]),
                false,
            ),
        )
        .unwrap();
        assert_eq!(f.cof, vec![2.5, 2.0, 1.5, 0.0]);
        assert_eq!(f.parameters, vec![2.5, 2.0, 1.5, 0.0]);
    }

    #[test]
    fn gstools_models_fit_their_own_variograms() {
        use crate::gstools::GsModel;
        let bins: Vec<f64> = (1..=24).map(|i| i as f64 * 1.5).collect();
        for (g, shape) in [
            (GsModel::Rational, Some(1.5)),
            (GsModel::SuperSpherical, Some(1.0)),
            (GsModel::Integral, Some(2.0)),
            (GsModel::TplExponential, Some(0.4)),
            (GsModel::TplSimple, Some(2.0)),
            (GsModel::Circular, None),
        ] {
            let model = Model::Gstools(g);
            let truth: Vec<f64> = match shape {
                Some(s) => vec![12.0, 1.0, s, 0.1],
                None => vec![12.0, 1.0, 0.1],
            };
            let experimental: Vec<f64> = bins.iter().map(|&h| value(model, h, &truth)).collect();
            let f = fit(&bins, &experimental, &opts(model, Method::Trf, true)).unwrap();
            for (got, want) in f.cof.iter().zip(&truth) {
                assert!(
                    (got - want).abs() < 1e-5 * want.abs().max(1.0),
                    "{g:?}: {:?} vs {truth:?}",
                    f.cof
                );
            }
        }
    }
}
