//! Spatial random fields by the randomisation method (`gstools.SRF`, `RandMeth`), seeded as
//! GSTools seeds them: numpy's legacy `RandomState` (MT19937 with `init_genrand`, the masked
//! bounded integers, the polar Box-Muller normal), a master stream handing out a seed for each
//! use.

use crate::kriging::{Model, Point};
use crate::numpy_random::Mt19937;
use crate::parallel::par_map;
use std::f64::consts::TAU;

/// Points per thread below which a field is summed on fewer threads.
const MIN_POINTS_PER_THREAD: usize = 16;

/// numpy's legacy `RandomState` seeded with an integer.
pub struct RandomState {
    mt: Mt19937,
    gauss: Option<f64>,
}

impl RandomState {
    pub fn new(seed: u32) -> Self {
        RandomState {
            mt: Mt19937::legacy(seed),
            gauss: None,
        }
    }

    /// `uniform(low, high)`: `low + (high - low) * u`.
    pub fn uniform(&mut self, low: f64, high: f64) -> f64 {
        low + (high - low) * self.mt.next_double()
    }

    /// `normal()`: numpy's `legacy_gauss`, the polar Box-Muller method keeping the second value.
    pub fn normal(&mut self) -> f64 {
        if let Some(g) = self.gauss.take() {
            return g;
        }
        loop {
            let x1 = 2.0 * self.mt.next_double() - 1.0;
            let x2 = 2.0 * self.mt.next_double() - 1.0;
            let r2 = x1 * x1 + x2 * x2;
            if r2 < 1.0 && r2 != 0.0 {
                let f = (-2.0 * r2.ln() / r2).sqrt();
                self.gauss = Some(f * x1);
                return f * x2;
            }
        }
    }
}

/// GSTools' `MasterRNG`: a `RandomState` handing out seeds by `randint(1, 2**16)`; each use of
/// GSTools' `RNG.random` is a new stream seeded with the next one.
pub struct Master(Mt19937);

impl Master {
    pub fn new(seed: u32) -> Self {
        Master(Mt19937::legacy(seed))
    }

    pub fn next_seed(&mut self) -> u32 {
        1 + self.0.interval(65_534)
    }

    pub fn stream(&mut self) -> RandomState {
        RandomState::new(self.next_seed())
    }
}

/// The random modes of the randomisation method: normal amplitudes and wave vectors drawn from
/// the model's spectral density (unused components 0).
pub struct Modes {
    pub z1: Vec<f64>,
    pub z2: Vec<f64>,
    pub k: Vec<Point>,
}

impl Modes {
    /// GSTools' `RandMeth` modes of a spherical, exponential or gaussian model of Geostatex's
    /// `range`, in 2D or 3D. The amplitudes come first (two streams). In 2D the exponential and
    /// gaussian wave vectors are GSTools' (a uniform angle, then the radius by inversion of the
    /// radial spectral CDF, a stream each); elsewhere GSTools samples by MCMC, and these are
    /// sampled exactly instead: GSTools' 3D directions (two streams) and the radius from the 3D
    /// radial spectrum (one stream), the third component dropped in 2D (a plane of a 3D field
    /// has the same covariance).
    pub fn new(
        model: Model,
        range: f64,
        dim: usize,
        mode_no: usize,
        master: &mut Master,
    ) -> Result<Self, String> {
        if !matches!(
            model,
            Model::Exponential | Model::Gaussian | Model::Spherical
        ) {
            return Err("simulation needs a spherical, exponential or gaussian model".to_string());
        }
        if !(dim == 2 || dim == 3) {
            return Err("simulation is in 2D or 3D".to_string());
        }
        let mut draw = |f: &mut dyn FnMut(&mut RandomState) -> f64| -> Vec<f64> {
            let mut s = master.stream();
            (0..mode_no).map(|_| f(&mut s)).collect()
        };
        let z1 = draw(&mut |s| s.normal());
        let z2 = draw(&mut |s| s.normal());
        let k = if dim == 2 && model != Model::Spherical {
            let angles = draw(&mut |s| s.uniform(0.0, TAU));
            let radii = draw(&mut |s| ppf_2d(model, range, s.uniform(0.0, 1.0)));
            angles
                .iter()
                .zip(&radii)
                .map(|(a, r)| [r * a.cos(), r * a.sin(), 0.0])
                .collect()
        } else {
            let azimuths = draw(&mut |s| s.uniform(0.0, TAU));
            let heights = draw(&mut |s| s.uniform(-1.0, 1.0));
            let radii = draw(&mut |s| radius_3d(model, range, s));
            (0..mode_no)
                .map(|j| {
                    let (a, z, r) = (azimuths[j], heights[j], radii[j]);
                    let c = (1.0 - z * z).sqrt();
                    let third = if dim == 3 { r * z } else { 0.0 };
                    [r * (c * a.cos()), r * (c * a.sin()), third]
                })
                .collect()
        };
        Ok(Modes { z1, z2, k })
    }

    /// `sum_j z1_j cos(k_j . x) + z2_j sin(k_j . x)` at each point, on its first `dim`
    /// coordinates, in GSTools' order (the phase summed by coordinate, the modes in order).
    pub fn summate(&self, points: &[Point], dim: usize) -> Vec<f64> {
        par_map(points.len(), MIN_POINTS_PER_THREAD, |i| {
            let x = points[i];
            let mut sum = 0.0;
            for (j, k) in self.k.iter().enumerate() {
                let mut phase = 0.0;
                for d in 0..dim {
                    phase += k[d] * x[d];
                }
                sum += self.z1[j] * phase.cos() + self.z2[j] * phase.sin();
            }
            sum
        })
    }
}

/// A field's model and generator: a spherical, exponential or gaussian model with Geostatex's
/// parameters, in `dim` dimensions, with `mode_no` modes.
#[derive(Clone, Copy, Debug)]
pub struct Spec {
    pub model: Model,
    pub psill: f64,
    pub range: f64,
    pub nugget: f64,
    pub dim: usize,
    pub mode_no: usize,
}

/// What a field is centred on: a constant mean (`gstools.SRF`), or the kriging estimate and
/// variance at each point (`gstools.CondSRF`).
pub enum Center<'a> {
    Mean(f64),
    Kriging {
        estimate: &'a [f64],
        variance: &'a [f64],
    },
}

/// One realisation at `points` (isometrised; the nugget noise follows their order), as GSTools
/// computes it from `seed`: the modes, the scaled sum `sqrt(psill / mode_no) * summate`, then,
/// with a nugget, the noise `sqrt(nugget) * normal` from the next stream. Unconditional:
/// `field + noise + mean`. Conditioned on kriging (variance `v`, clamped at 0):
/// `estimate + sqrt(max(v - nugget, 0) / psill) * field + sqrt(min(v, nugget) / nugget) * noise`,
/// so the data are reproduced where `v` is 0 and the pointwise mean and variance are the
/// kriging ones.
pub fn field(spec: &Spec, points: &[Point], seed: u32, center: Center) -> Result<Vec<f64>, String> {
    if spec.mode_no == 0 {
        return Err("mode_no must be positive".to_string());
    }
    if let Center::Kriging { estimate, variance } = center {
        if estimate.len() != points.len() || variance.len() != points.len() {
            return Err("the kriging estimate and variance need one value per point".to_string());
        }
    }
    let mut master = Master::new(seed);
    let modes = Modes::new(spec.model, spec.range, spec.dim, spec.mode_no, &mut master)?;
    let scale = (spec.psill / spec.mode_no as f64).sqrt();
    let raw: Vec<f64> = modes
        .summate(points, spec.dim)
        .into_iter()
        .map(|s| scale * s)
        .collect();
    let noise: Option<Vec<f64>> = (spec.nugget > 0.0).then(|| {
        let (mut stream, amplitude) = (master.stream(), spec.nugget.sqrt());
        (0..points.len())
            .map(|_| amplitude * stream.normal())
            .collect()
    });
    let noise_at = |i: usize| noise.as_ref().map_or(0.0, |n| n[i]);
    Ok(match center {
        Center::Mean(mean) => (0..points.len())
            .map(|i| raw[i] + noise_at(i) + mean)
            .collect(),
        Center::Kriging { estimate, variance } => (0..points.len())
            .map(|i| {
                let v = variance[i].max(0.0);
                if spec.nugget > 0.0 {
                    let field_var = (v - spec.nugget).max(0.0);
                    let noise_scale = ((v - field_var) / spec.nugget).sqrt();
                    estimate[i]
                        + (field_var / spec.psill).sqrt() * raw[i]
                        + noise_scale * noise_at(i)
                } else {
                    estimate[i] + (v / spec.psill).sqrt() * raw[i]
                }
            })
            .collect(),
    })
}

/// GSTools' 2D radial spectral ppf (`Exponential` and `Gaussian` with `rescale=1`).
fn ppf_2d(model: Model, range: f64, u: f64) -> f64 {
    if model == Model::Exponential {
        let len = range / 3.0;
        // `np.divide(1, u**2, where=not isclose(u, 0))`, infinity otherwise.
        let u_power = if u.abs() <= 1e-8 {
            f64::INFINITY
        } else {
            1.0 / (u * u)
        };
        (u_power - 1.0).sqrt() / len
    } else {
        let len = range * 4.0 / 7.0;
        2.0 / len * (-(1.0 - u).ln()).sqrt()
    }
}

/// A radius from the 3D radial spectral density of the model (`range` Geostatex's).
fn radius_3d(model: Model, range: f64, s: &mut RandomState) -> f64 {
    match model {
        // Density `exp(-(k len / 2)^2)`: each component normal with variance `2 / len^2`.
        Model::Gaussian => {
            let len = range * 4.0 / 7.0;
            let (a, b, c) = (s.normal(), s.normal(), s.normal());
            std::f64::consts::SQRT_2 / len * (a * a + b * b + c * c).sqrt()
        }
        // CDF `2 / pi (atan(t) - t / (1 + t^2))`, `t = k len = tan(theta)`:
        // `theta - sin(2 theta) / 2 = u pi / 2`, solved by bisection.
        Model::Exponential => {
            let len = range / 3.0;
            let target = s.uniform(0.0, 1.0) * std::f64::consts::FRAC_PI_2;
            let (mut lo, mut hi) = (0.0, std::f64::consts::FRAC_PI_2);
            for _ in 0..64 {
                let mid = 0.5 * (lo + hi);
                if mid - (2.0 * mid).sin() / 2.0 < target {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            (0.5 * (lo + hi)).tan() / len
        }
        // Density of `x = k range / 2` proportional to `j1(x)^2` (the spectrum of the overlap
        // of two balls), by rejection under `x^2 / 9` up to `sqrt(6)` and `4 / x^2` beyond.
        _ => {
            let t0 = 6.0_f64.sqrt();
            let (a1, a2) = (t0.powi(3) / 27.0, 4.0 / t0);
            loop {
                let u1 = s.uniform(0.0, 1.0) * (a1 + a2);
                let u2 = s.uniform(0.0, 1.0);
                let (x, envelope) = if u1 < a1 {
                    let x = t0 * (u1 / a1).cbrt();
                    (x, x * x / 9.0)
                } else {
                    let x = t0 / (1.0 - (u1 - a1) / a2);
                    (x, 4.0 / (x * x))
                };
                if u2 * envelope <= j1(x).powi(2) {
                    return 2.0 * x / range;
                }
            }
        }
    }
}

/// The spherical Bessel function `j1(x) = (sin x - x cos x) / x^2` (its series near 0).
fn j1(x: f64) -> f64 {
    if x < 1e-3 {
        x / 3.0 - x.powi(3) / 30.0 + x.powi(5) / 840.0
    } else {
        (x.sin() - x * x.cos()) / (x * x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Recorded from numpy 2.5 / GSTools 1.7 (test/reference/fixtures/simulation/srf.json).

    #[test]
    fn legacy_random_state_draws_as_numpy() {
        let mut r = RandomState::new(0);
        let normal: Vec<f64> = (0..4).map(|_| r.normal()).collect();
        assert_eq!(
            normal,
            [
                1.764052345967664,
                0.4001572083672233,
                0.9787379841057392,
                2.240893199201458
            ]
        );
        let mut r = RandomState::new(19);
        let uniform: Vec<f64> = (0..3).map(|_| r.uniform(0.0, TAU)).collect();
        assert_eq!(
            uniform,
            [0.6128216934684594, 4.78307303490608, 1.551557044783373]
        );
        let mut r = RandomState::new(4294967295);
        let rand: Vec<f64> = (0..3).map(|_| r.uniform(0.0, 1.0)).collect();
        assert_eq!(
            rand,
            [0.0976320289940138, 0.9123828453026218, 0.78903530185164]
        );
        let mut r = RandomState::new(4294967295);
        assert_eq!(r.normal(), 0.6484086742306527);
    }

    #[test]
    fn master_seeds_as_gstools() {
        for (seed, want) in [
            (0, [2733, 43568, 42614, 52417, 45892, 21244, 30404, 32104]),
            (19, [63070, 10863, 17142, 34147, 21417, 33812, 39221, 17739]),
            (
                4294967295,
                [27044, 16675, 60429, 44104, 14261, 65089, 20695, 507],
            ),
        ] {
            let mut master = Master::new(seed);
            let got: Vec<u32> = (0..8).map(|_| master.next_seed()).collect();
            assert_eq!(got, want, "seed {seed}");
        }
    }

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * b.abs().max(1.0)
    }

    #[test]
    fn modes_and_field_as_gstools_in_2d() {
        let at = [
            [53.314066863144724, 4.300413908129263, 0.0],
            [25.967925319138875, 74.36639122731935, 0.0],
        ];
        for (model, k, field) in [
            (
                Model::Exponential,
                [
                    [-0.47888331201222983, 0.25624403500803816],
                    [-0.05552359922465914, 0.14714413789915415],
                ],
                [-1.127646933133349, -1.8741665644151497],
            ),
            (
                Model::Gaussian,
                [
                    [-0.045975784461033016, 0.02460102540105656],
                    [-0.03612119849444804, 0.09572546964839546],
                ],
                [-0.831399137485737, -0.42826743614880336],
            ),
        ] {
            let modes = Modes::new(model, 30.0, 2, 100, &mut Master::new(19)).unwrap();
            assert_eq!(modes.z1[..2], [-0.00859126002745369, 0.6026666289294729]);
            assert_eq!(modes.z2[..2], [-0.3412194903201838, 1.6716976713168386]);
            for (j, want) in k.iter().enumerate() {
                for (d, w) in want.iter().enumerate() {
                    let got = modes.k[j][d];
                    assert!(close(got, *w, 1e-15), "{model:?} k[{j}][{d}]: {got} vs {w}");
                }
            }
            let raw = modes.summate(&at, 2);
            for (t, want) in field.iter().enumerate() {
                let got = (1.0_f64 / 100.0).sqrt() * raw[t];
                assert!(
                    close(got, *want, 1e-12),
                    "{model:?} field {t}: {got} vs {want}"
                );
            }
        }
    }

    /// The correlation of each model at lag `h` (`range` is Geostatex's, PyKrige's).
    fn correlation(model: Model, range: f64, h: f64) -> f64 {
        match model {
            Model::Exponential => (-3.0 * h / range).exp(),
            Model::Gaussian => (-(h / (range * 4.0 / 7.0)).powi(2)).exp(),
            _ => {
                let r = (h / range).min(1.0);
                1.0 - 1.5 * r + 0.5 * r.powi(3)
            }
        }
    }

    #[test]
    fn the_wave_vectors_reproduce_each_model_s_correlation() {
        // Bochner: E[cos(k . h)] over the spectral distribution is the correlation at h.
        let n = 200_000;
        for model in [Model::Exponential, Model::Gaussian, Model::Spherical] {
            for dim in [2, 3] {
                let modes = Modes::new(model, 10.0, dim, n, &mut Master::new(7)).unwrap();
                for h in [1.0, 3.0, 6.0, 12.0] {
                    for dir in [[1.0, 0.0, 0.0], [0.6, 0.8, 0.0], [0.0, 0.6, 0.8]] {
                        if dim == 2 && dir[2] != 0.0 {
                            continue;
                        }
                        let mean = modes
                            .k
                            .iter()
                            .map(|k| (h * (k[0] * dir[0] + k[1] * dir[1] + k[2] * dir[2])).cos())
                            .sum::<f64>()
                            / n as f64;
                        let want = correlation(model, 10.0, h);
                        assert!(
                            (mean - want).abs() < 0.01,
                            "{model:?} {dim}D h {h} dir {dir:?}: {mean} vs {want}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn the_spherical_spectrum_is_a_density() {
        // The rejection sampler's target: j1(x)^2 integrates to pi / 6 over (0, inf).
        let (mut sum, dx) = (0.0, 1e-3);
        let mut x = dx / 2.0;
        while x < 2000.0 {
            sum += j1(x).powi(2) * dx;
            x += dx;
        }
        // The tail beyond 2000 adds about 1 / (2 * 2000).
        assert!(
            (sum + 1.0 / 4000.0 - std::f64::consts::PI / 6.0).abs() < 1e-4,
            "{sum}"
        );
        assert!(j1(1e-6) > 0.0 && (j1(1e-6) - 1e-6 / 3.0).abs() < 1e-18);
    }

    #[test]
    fn a_field_adds_the_mean_and_the_nugget_or_conditions_on_the_kriging() {
        let points = [[1.0, 2.0, 0.0], [3.0, -1.0, 0.0], [0.5, 0.5, 0.0]];
        let spec = Spec {
            model: Model::Exponential,
            psill: 0.8,
            range: 10.0,
            nugget: 0.2,
            dim: 2,
            mode_no: 50,
        };
        // By hand: the modes, then one more stream for the nugget noise.
        let mut master = Master::new(7);
        let modes = Modes::new(Model::Exponential, 10.0, 2, 50, &mut master).unwrap();
        let raw: Vec<f64> = modes
            .summate(&points, 2)
            .iter()
            .map(|s| (0.8_f64 / 50.0).sqrt() * s)
            .collect();
        let mut stream = master.stream();
        let noise: Vec<f64> = (0..3).map(|_| 0.2_f64.sqrt() * stream.normal()).collect();

        let f = field(&spec, &points, 7, Center::Mean(1.5)).unwrap();
        for t in 0..3 {
            assert_eq!(f[t], raw[t] + noise[t] + 1.5);
        }

        // Conditioned: no field where the kriging variance is 0, the nugget noise alone where it
        // is the nugget, both scaled beyond.
        let (estimate, variance) = ([4.0, 5.0, 6.0], [-1e-17, 0.2, 0.5]);
        let c = field(
            &spec,
            &points,
            7,
            Center::Kriging {
                estimate: &estimate,
                variance: &variance,
            },
        )
        .unwrap();
        assert_eq!(c[0], 4.0);
        assert_eq!(c[1], 5.0 + noise[1]);
        assert_eq!(c[2], 6.0 + (0.3_f64 / 0.8).sqrt() * raw[2] + noise[2]);

        let none = Spec {
            nugget: 0.0,
            ..spec
        };
        let c = field(
            &none,
            &points,
            7,
            Center::Kriging {
                estimate: &estimate,
                variance: &variance,
            },
        )
        .unwrap();
        assert_eq!(c[2], 6.0 + (0.5_f64 / 0.8).sqrt() * raw[2]);
        let short = Center::Kriging {
            estimate: &estimate[..2],
            variance: &variance,
        };
        assert!(field(&none, &points, 7, short).is_err());
    }

    #[test]
    fn models_without_a_spectrum_are_refused() {
        assert!(Modes::new(Model::Linear, 1.0, 2, 10, &mut Master::new(1)).is_err());
        assert!(Modes::new(Model::Exponential, 1.0, 1, 10, &mut Master::new(1)).is_err());
    }
}
