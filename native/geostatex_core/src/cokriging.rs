//! Ordinary cokriging with a linear model of coregionalisation (R `gstat`'s multivariable
//! prediction with a constant mean per variable), and the LMC fit (`gstat::fit.lmc`).

pub use crate::kriging::Point;
use crate::memory;
use crate::parallel::par_map;
use faer::linalg::solvers::Solve;
use faer::{Mat, Side};

/// Targets per thread below which a prediction runs on fewer threads.
const MIN_TARGETS_PER_THREAD: usize = 16;

/// A structure of a linear model of coregionalisation (Geostatex's, PyKrige's, ranges).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Kind {
    Nugget,
    Spherical,
    Exponential,
    Gaussian,
}

impl Kind {
    /// The structure's unit variogram at distance `h` (0 at 0, 1 at its sill).
    pub fn unit(self, range: f64, h: f64) -> f64 {
        match self {
            Kind::Nugget => {
                if h == 0.0 {
                    0.0
                } else {
                    1.0
                }
            }
            Kind::Spherical => {
                if h >= range {
                    1.0
                } else {
                    let r = h / range;
                    1.5 * r - 0.5 * r.powi(3)
                }
            }
            Kind::Exponential => 1.0 - (-h / (range / 3.0)).exp(),
            Kind::Gaussian => 1.0 - (-(h / (range * 4.0 / 7.0)).powi(2)).exp(),
        }
    }
}

/// One structure and its matrix of partial sills (`n x n`, row-major).
#[derive(Clone, Debug, PartialEq)]
pub struct Structure {
    pub kind: Kind,
    pub range: f64,
    pub sills: Vec<f64>,
}

/// A linear model of coregionalisation of `n` variables: every direct and cross variogram is
/// `sum_c sills_c[a, b] * unit_c(h)`, each sill matrix symmetric and positive semi-definite.
#[derive(Clone, Debug, PartialEq)]
pub struct Lmc {
    pub n: usize,
    pub structures: Vec<Structure>,
}

impl Lmc {
    pub fn new(n: usize, structures: Vec<Structure>) -> Result<Self, String> {
        if n == 0 || structures.is_empty() {
            return Err("a model of coregionalisation needs variables and structures".to_string());
        }
        for s in &structures {
            if s.sills.len() != n * n {
                return Err(format!("every sill matrix must be {n} x {n}"));
            }
            if s.kind != Kind::Nugget
                && s.range.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater)
            {
                return Err("every structure but the nugget needs a positive range".to_string());
            }
            let scale = s.sills.iter().fold(0.0_f64, |m, v| m.max(v.abs())).max(1.0);
            for a in 0..n {
                for b in 0..a {
                    if (s.sills[a * n + b] - s.sills[b * n + a]).abs() > 1e-12 * scale {
                        return Err("every sill matrix must be symmetric".to_string());
                    }
                }
            }
            if eigenvalues(&s.sills, n)?
                .iter()
                .any(|&e| e < -1e-10 * scale)
            {
                return Err(
                    "every sill matrix must be positive semi-definite (a valid model of \
                     coregionalisation)"
                        .to_string(),
                );
            }
        }
        Ok(Lmc { n, structures })
    }

    /// The covariance of variables `a` and `b` at distance `h`: `sum_c sills_c[a, b] (1 - unit_c)`.
    pub fn covariance(&self, a: usize, b: usize, h: f64) -> f64 {
        self.structures
            .iter()
            .map(|s| s.sills[a * self.n + b] * (1.0 - s.kind.unit(s.range, h)))
            .sum()
    }
}

/// The eigenvalues of a symmetric `n x n` matrix (row-major).
fn eigenvalues(m: &[f64], n: usize) -> Result<Vec<f64>, String> {
    let a = Mat::from_fn(n, n, |i, j| m[i * n + j]);
    let eig = a
        .self_adjoint_eigen(Side::Lower)
        .map_err(|e| format!("eigendecomposition failed: {e:?}"))?;
    let s = eig.S();
    Ok((0..n).map(|i| s[i]).collect())
}

fn distance(a: Point, b: Point) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// Ordinary cokriging of `n` variables, each with its points and values (isotopic or not).
pub struct Cokriging {
    pub points: Vec<Vec<Point>>,
    pub values: Vec<Vec<f64>>,
    pub lmc: Lmc,
}

impl Cokriging {
    /// Each variable's estimate at `targets` and, per target, the error covariance matrix of the
    /// estimates (`n x n`, row-major; its diagonal the cokriging variances). The system is the
    /// covariances between the observations bordered by one unbiasedness constraint per
    /// variable; predicting variable `k` takes the covariances with `k` at the target and the
    /// `k`-th constraint, and the error covariances are `C(0) - X^T B`.
    #[allow(clippy::type_complexity)]
    pub fn predict(&self, targets: &[Point]) -> Result<(Vec<Vec<f64>>, Vec<Vec<f64>>), String> {
        let v = self.lmc.n;
        if self.points.len() != v || self.values.len() != v {
            return Err(format!("the model has {v} variables"));
        }
        if self
            .points
            .iter()
            .zip(&self.values)
            .any(|(p, z)| p.is_empty() || p.len() != z.len())
        {
            return Err("every variable needs points with one value each".to_string());
        }
        if targets.is_empty() {
            return Ok((vec![Vec::new(); v], Vec::new()));
        }
        let obs: Vec<(usize, Point)> = self
            .points
            .iter()
            .enumerate()
            .flat_map(|(k, ps)| ps.iter().map(move |&p| (k, p)))
            .collect();
        let z: Vec<f64> = self.values.concat();
        let (n, size) = (obs.len(), obs.len() + v);
        memory::ensure(memory::system_bytes(size, false), || {
            format!("cokriging of {n} observations")
        })?;
        let cov = |a: usize, b: usize, h: f64| self.lmc.covariance(a, b, h);
        let a = Mat::from_fn(size, size, |i, j| match (i < n, j < n) {
            (true, true) => cov(obs[i].0, obs[j].0, distance(obs[i].1, obs[j].1)),
            (true, false) => f64::from(u8::from(obs[i].0 == j - n)),
            (false, true) => f64::from(u8::from(obs[j].0 == i - n)),
            (false, false) => 0.0,
        });
        let lu = a.partial_piv_lu();
        let c0: Vec<f64> = (0..v * v).map(|kl| cov(kl / v, kl % v, 0.0)).collect();
        let rows = par_map(targets.len(), MIN_TARGETS_PER_THREAD, |t| {
            let p = targets[t];
            let b = Mat::from_fn(size, v, |i, k| {
                if i < n {
                    cov(obs[i].0, k, distance(obs[i].1, p))
                } else {
                    f64::from(u8::from(i - n == k))
                }
            });
            let x = lu.solve(&b);
            let estimates: Vec<f64> = (0..v)
                .map(|k| (0..n).map(|i| x[(i, k)] * z[i]).sum())
                .collect();
            let covariance: Vec<f64> = (0..v * v)
                .map(|kl| {
                    let (k, l) = (kl / v, kl % v);
                    c0[kl] - (0..n).map(|i| x[(i, k)] * b[(i, l)]).sum::<f64>() - x[(n + l, k)]
                })
                .collect();
            (estimates, covariance)
        });
        if rows
            .iter()
            .any(|(e, c)| e.iter().chain(c).any(|x| !x.is_finite()))
        {
            return Err("the cokriging matrix is singular".to_string());
        }
        let mut estimates = vec![Vec::with_capacity(targets.len()); v];
        let mut covariances = Vec::with_capacity(targets.len());
        for (e, c) in rows {
            for (k, value) in e.into_iter().enumerate() {
                estimates[k].push(value);
            }
            covariances.push(c);
        }
        Ok((estimates, covariances))
    }
}

/// A sample (direct or cross) variogram: lag distances, semivariances and pair counts.
#[derive(Clone, Debug, PartialEq)]
pub struct Sample {
    pub dist: Vec<f64>,
    pub gamma: Vec<f64>,
    pub np: Vec<f64>,
}

/// gstat's `fit.lmc` with the ranges fixed: each direct and cross sample variogram (`(a, b,
/// sample)`, every pair `a <= b` once) gets the partial sills of `kinds` by weighted least squares
/// (weights `np / dist^2`, `fit.variogram`'s default `fit.method = 7`); then each structure's sill
/// matrix becomes its closest positive semi-definite matrix (negative eigenvalues set to 0) and
/// its diagonal is multiplied by `correct_diagonal`.
pub fn fit_lmc(
    n: usize,
    samples: &[(usize, usize, Sample)],
    kinds: &[(Kind, f64)],
    correct_diagonal: f64,
) -> Result<Lmc, String> {
    let m = kinds.len();
    if m == 0 {
        return Err("the model needs at least one structure".to_string());
    }
    let mut sills = vec![vec![f64::NAN; n * n]; m];
    for (a, b, s) in samples {
        let (a, b) = ((*a).min(*b), (*a).max(*b));
        if b >= n || !sills[0][a * n + b].is_nan() {
            return Err(format!(
                "sample variogram ({a}, {b}) is out of range or repeated"
            ));
        }
        if s.dist.len() != s.gamma.len() || s.dist.len() != s.np.len() || s.dist.len() < m {
            return Err(format!(
                "sample variogram ({a}, {b}) needs {m} lags or more"
            ));
        }
        let x = Mat::from_fn(m, m, |r, c| {
            (0..s.dist.len())
                .map(|j| {
                    let (h, w) = (s.dist[j], s.np[j] / s.dist[j].powi(2));
                    w * kinds[r].0.unit(kinds[r].1, h) * kinds[c].0.unit(kinds[c].1, h)
                })
                .sum::<f64>()
        });
        let y = Mat::from_fn(m, 1, |r, _| {
            (0..s.dist.len())
                .map(|j| {
                    let (h, w) = (s.dist[j], s.np[j] / s.dist[j].powi(2));
                    w * kinds[r].0.unit(kinds[r].1, h) * s.gamma[j]
                })
                .sum::<f64>()
        });
        let fit = x.partial_piv_lu().solve(&y);
        for (c, sill) in sills.iter_mut().enumerate() {
            if !fit[(c, 0)].is_finite() {
                return Err(format!("sample variogram ({a}, {b}) cannot fix the sills"));
            }
            sill[a * n + b] = fit[(c, 0)];
            sill[b * n + a] = fit[(c, 0)];
        }
    }
    if sills[0].iter().any(|v| v.is_nan()) {
        return Err("every direct and cross sample variogram is needed".to_string());
    }
    let structures = kinds
        .iter()
        .zip(sills)
        .map(|(&(kind, range), s)| {
            let mut s = closest_psd(&s, n)?;
            for a in 0..n {
                s[a * n + a] *= correct_diagonal;
            }
            Ok(Structure {
                kind,
                range,
                sills: s,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    Lmc::new(n, structures)
}

/// The closest positive semi-definite matrix: `U max(D, 0) U^T`.
fn closest_psd(m: &[f64], n: usize) -> Result<Vec<f64>, String> {
    let a = Mat::from_fn(n, n, |i, j| m[i * n + j]);
    let eig = a
        .self_adjoint_eigen(Side::Lower)
        .map_err(|e| format!("eigendecomposition failed: {e:?}"))?;
    let (u, d) = (eig.U(), eig.S());
    Ok((0..n * n)
        .map(|ij| {
            let (i, j) = (ij / n, ij % n);
            (0..n).map(|k| u[(i, k)] * d[k].max(0.0) * u[(j, k)]).sum()
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kriging::{Inverse, Kriging, Model};

    fn line_points() -> Vec<Point> {
        (0..6)
            .map(|i| [i as f64 * 3.0, (i % 2) as f64 * 2.0, 0.0])
            .collect()
    }

    fn ordinary(points: Vec<Point>, values: Vec<f64>, psill: f64, nugget: f64) -> Kriging {
        Kriging {
            points,
            values,
            model: Model::Spherical,
            params: vec![psill, 10.0, nugget],
            geographic: false,
            exact_values: true,
            inverse: Inverse::Exact,
            center: [0.0; 3],
            scaling: [1.0, 1.0],
            angles: [0.0; 3],
            drifts: Vec::new(),
            data_drift: Vec::new(),
            gamma_table: None,
            simple_mean: None,
        }
    }

    fn lmc(nugget: Vec<f64>, sph: Vec<f64>, n: usize) -> Lmc {
        Lmc::new(
            n,
            vec![
                Structure {
                    kind: Kind::Nugget,
                    range: 0.0,
                    sills: nugget,
                },
                Structure {
                    kind: Kind::Spherical,
                    range: 10.0,
                    sills: sph,
                },
            ],
        )
        .unwrap()
    }

    #[test]
    fn one_variable_is_ordinary_kriging() {
        let values = vec![1.0, 2.5, 2.0, 3.5, 3.0, 4.0];
        let ck = Cokriging {
            points: vec![line_points()],
            values: vec![values.clone()],
            lmc: lmc(vec![0.1], vec![0.8], 1),
        };
        let targets = [[4.0, 1.0, 0.0], [6.0, 0.0, 0.0], [40.0, 5.0, 0.0]];
        let (est, cov) = ck.predict(&targets).unwrap();
        let ok = ordinary(line_points(), values, 0.8, 0.1);
        let (z, s) = ok.execute(&targets, &[], &[false; 3], None).unwrap();
        for t in 0..3 {
            assert!(
                (est[0][t] - z[t]).abs() < 1e-12,
                "target {t}: {} vs {}",
                est[0][t],
                z[t]
            );
            assert!(
                (cov[t][0] - s[t]).abs() < 1e-12,
                "target {t}: {} vs {}",
                cov[t][0],
                s[t]
            );
        }
        // At a datum: its value, variance 0.
        assert!((est[0][1] - 2.0).abs() < 1e-12 && cov[1][0].abs() < 1e-12);
    }

    #[test]
    fn uncorrelated_variables_are_kriged_apart() {
        let (a, b) = (
            vec![1.0, 2.5, 2.0, 3.5, 3.0, 4.0],
            vec![7.0, 6.0, 6.5, 5.0, 5.5],
        );
        let pb: Vec<Point> = line_points()
            .iter()
            .take(5)
            .map(|p| [p[0] + 1.0, p[1] + 0.5, 0.0])
            .collect();
        let ck = Cokriging {
            points: vec![line_points(), pb.clone()],
            values: vec![a.clone(), b.clone()],
            lmc: lmc(vec![0.1, 0.0, 0.0, 0.2], vec![0.8, 0.0, 0.0, 0.5], 2),
        };
        let targets = [[4.0, 1.0, 0.0], [10.0, 0.5, 0.0]];
        let (est, cov) = ck.predict(&targets).unwrap();
        for (v, (points, values, psill, nugget)) in
            [(line_points(), a, 0.8, 0.1), (pb, b, 0.5, 0.2)]
                .into_iter()
                .enumerate()
        {
            let (z, s) = ordinary(points, values, psill, nugget)
                .execute(&targets, &[], &[false; 2], None)
                .unwrap();
            for t in 0..2 {
                assert!((est[v][t] - z[t]).abs() < 1e-12, "variable {v} target {t}");
                assert!(
                    (cov[t][v * 2 + v] - s[t]).abs() < 1e-12,
                    "variable {v} target {t}"
                );
                assert!(cov[t][1].abs() < 1e-12 && cov[t][2].abs() < 1e-12);
            }
        }
    }

    fn sample(sills: [f64; 2], noise: f64) -> Sample {
        let dist: Vec<f64> = (1..=12).map(|i| i as f64 * 1.5).collect();
        let gamma = dist
            .iter()
            .enumerate()
            .map(|(j, &h)| {
                sills[0] * Kind::Nugget.unit(0.0, h)
                    + sills[1] * Kind::Spherical.unit(10.0, h)
                    + noise * if j % 2 == 0 { 1.0 } else { -1.0 }
            })
            .collect();
        Sample {
            dist,
            gamma,
            np: (1..=12).map(|i| 100.0 + i as f64).collect(),
        }
    }

    #[test]
    fn the_lmc_fit_recovers_exact_sills_and_projects_onto_valid_ones() {
        let kinds = [(Kind::Nugget, 0.0), (Kind::Spherical, 10.0)];
        let samples = vec![
            (0, 0, sample([0.1, 0.8], 0.0)),
            (0, 1, sample([0.03, 0.4], 0.0)),
            (1, 1, sample([0.05, 0.5], 0.0)),
        ];
        let lmc = fit_lmc(2, &samples, &kinds, 1.0).unwrap();
        let want = [[0.1, 0.03, 0.03, 0.05], [0.8, 0.4, 0.4, 0.5]];
        for (s, w) in lmc.structures.iter().zip(want) {
            for (a, b) in s.sills.iter().zip(w) {
                assert!((a - b).abs() < 1e-12, "{:?} vs {w:?}", s.sills);
            }
        }
        // A cross sill too large for its direct ones: the closest positive semi-definite matrix
        // (eigenvalues of [[1, 2], [2, 1]] are 3 and -1: keep 3 * [1, 1; 1, 1] / 2), then the
        // diagonal correction.
        let samples = vec![
            (0, 0, sample([0.0, 1.0], 0.0)),
            (0, 1, sample([0.0, 2.0], 0.0)),
            (1, 1, sample([0.0, 1.0], 0.0)),
        ];
        let lmc = fit_lmc(2, &samples, &kinds, 1.01).unwrap();
        let s = &lmc.structures[1].sills;
        for (a, b) in s.iter().zip([1.5 * 1.01, 1.5, 1.5, 1.5 * 1.01]) {
            assert!((a - b).abs() < 1e-12, "{s:?}");
        }
        // Weighted least squares with noise: the normal equations by hand.
        let noisy = sample([0.1, 0.8], 0.02);
        let lmc = fit_lmc(1, &[(0, 0, noisy.clone())], &kinds, 1.0).unwrap();
        let (mut xtx, mut xty) = ([[0.0; 2]; 2], [0.0; 2]);
        for j in 0..noisy.dist.len() {
            let w = noisy.np[j] / noisy.dist[j].powi(2);
            let x = [1.0, Kind::Spherical.unit(10.0, noisy.dist[j])];
            for r in 0..2 {
                xty[r] += w * x[r] * noisy.gamma[j];
                for c in 0..2 {
                    xtx[r][c] += w * x[r] * x[c];
                }
            }
        }
        let det = xtx[0][0] * xtx[1][1] - xtx[0][1] * xtx[1][0];
        let nugget = (xty[0] * xtx[1][1] - xtx[0][1] * xty[1]) / det;
        let psill = (xtx[0][0] * xty[1] - xtx[1][0] * xty[0]) / det;
        assert!((lmc.structures[0].sills[0] - nugget).abs() < 1e-12);
        assert!((lmc.structures[1].sills[0] - psill).abs() < 1e-12);
        assert!(
            fit_lmc(2, &samples[..2], &kinds, 1.0).is_err(),
            "a missing pair"
        );
    }

    #[test]
    fn a_model_of_coregionalisation_must_be_valid() {
        let structure = |sills: Vec<f64>| Structure {
            kind: Kind::Spherical,
            range: 10.0,
            sills,
        };
        // Not positive semi-definite, not symmetric, wrong size.
        assert!(Lmc::new(2, vec![structure(vec![1.0, 2.0, 2.0, 1.0])]).is_err());
        assert!(Lmc::new(2, vec![structure(vec![1.0, 0.5, 0.4, 1.0])]).is_err());
        assert!(Lmc::new(2, vec![structure(vec![1.0, 0.5, 0.5])]).is_err());
        assert!(Lmc::new(2, vec![structure(vec![1.0, 1.0, 1.0, 1.0])]).is_ok());
    }
}
