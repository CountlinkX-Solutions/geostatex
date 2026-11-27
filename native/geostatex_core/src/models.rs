//! Theoretical variogram models with scikit-gstat's parametrisation: effective range `r`, sill
//! `c0`, nugget `b`, and the shape `s` of the stable and Matérn models.

/// A variogram model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Model {
    Spherical,
    Exponential,
    Gaussian,
    Cubic,
    Stable,
    Matern,
    /// A GSTools model with at most one shape (`r` its length scale, `rescale = 1`).
    Gstools(crate::gstools::GsModel),
}

impl Model {
    /// Whether the model has a shape parameter (stable, Matérn, most GSTools models).
    pub fn has_shape(self) -> bool {
        match self {
            Model::Gstools(g) => g.shapes() == 1,
            _ => matches!(self, Model::Stable | Model::Matern),
        }
    }

    /// The upper bound of the shape in fits (scikit-gstat's, or GSTools' default bounds).
    pub fn shape_bound(self) -> f64 {
        match self {
            Model::Matern => 20.0,
            Model::Gstools(g) => g.shape_bounds().1,
            _ => 2.0,
        }
    }

    /// The lower bound of the shape in fits (0, or GSTools' default bounds).
    pub fn shape_lower(self) -> f64 {
        match self {
            Model::Gstools(g) => g.shape_bounds().0,
            _ => 0.0,
        }
    }
}

/// Semivariance at lag `h`. `cof` is `[r, c0, b]`, or `[r, c0, s, b]` with a shape.
pub fn value(model: Model, h: f64, cof: &[f64]) -> f64 {
    let (r, c0) = (cof[0], cof[1]);
    let b = cof[cof.len() - 1];
    match model {
        Model::Spherical => {
            let a = r / 1.0;
            if h <= r {
                b + c0 * (1.5 * (h / a) - 0.5 * (h / a).powf(3.0))
            } else {
                b + c0
            }
        }
        Model::Exponential => {
            let a = r / 3.0;
            b + c0 * (1.0 - (-(h / a)).exp())
        }
        Model::Gaussian => {
            let a = r / 2.0;
            b + c0 * (1.0 - (-(h.powi(2) / a.powi(2))).exp())
        }
        Model::Cubic => {
            let a = r / 1.0;
            if h < r {
                b + c0
                    * (7.0 * (h.powi(2) / a.powi(2)) - 35.0 / 4.0 * (h.powi(3) / a.powi(3))
                        + 7.0 / 2.0 * (h.powi(5) / a.powi(5))
                        - 3.0 / 4.0 * (h.powi(7) / a.powi(7)))
            } else {
                b + c0
            }
        }
        Model::Stable => {
            if h == 0.0 {
                return b;
            }
            let s = cof[2];
            let a = r / 3f64.powf(1.0 / s);
            b + c0 * (1.0 - (-(h / a).powf(s)).exp())
        }
        Model::Gstools(g) => b + c0 * (1.0 - crate::gstools::cor(g, h / r, &cof[2..cof.len() - 1])),
        Model::Matern => {
            if h == 0.0 {
                return b;
            }
            let s = cof[2];
            let a = r / 2.0;
            let y = h * s.sqrt() / a;
            b + c0 * (1.0 - 2.0 / libm::tgamma(s) * y.powf(s) * bessel_k(s, 2.0 * y))
        }
    }
}

/// The modified Bessel function of the second kind `K_nu(x)` for real `nu` and `x > 0`
/// (`scipy.special.kv`), from `K_nu(x) = int_0^inf exp(-x cosh t) cosh(nu t) dt` with the
/// trapezoidal rule, whose error falls like `exp(-pi^2 / step)` for this integrand.
pub fn bessel_k(nu: f64, x: f64) -> f64 {
    if x <= 0.0 || x.is_nan() {
        return f64::NAN;
    }
    if x > 745.0 {
        return 0.0;
    }
    let nu = nu.abs();
    let step = 0.02;
    // log of the integrand, computed without overflowing cosh(nu t).
    let log_f = |t: f64| -x * t.cosh() + nu * t + (0.5 * (1.0 + (-2.0 * nu * t).exp())).ln();
    let peak = (nu / x).asinh();
    let top = log_f(peak);
    let mut sum = 0.5 * log_f(0.0).exp();
    let mut k = 1.0;
    loop {
        let t = k * step;
        let lf = log_f(t);
        sum += lf.exp();
        if t > peak && lf < top - 40.0 {
            break;
        }
        k += 1.0;
    }
    sum * step
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bessel_k_matches_known_values() {
        // scipy.special.kv values.
        let cases = [
            (0.5, 1.0, 0.4610685044478946),
            (1.0, 2.0, 0.13986588181652246),
            (2.5, 0.1, 1187.0212236418931),
            (20.0, 3.0, 16254643952204.332),
            (0.3, 50.0, 3.413208199536853e-23),
        ];
        for (nu, x, want) in cases {
            let got = bessel_k(nu, x);
            assert!(
                ((got - want) / want).abs() < 1e-12,
                "K_{nu}({x}) = {got}, want {want}"
            );
        }
    }

    #[test]
    fn models_reach_the_nugget_at_zero_and_the_sill_at_the_range() {
        let cof = [10.0, 2.0, 0.5];
        assert_eq!(value(Model::Spherical, 0.0, &cof), 0.5);
        assert_eq!(value(Model::Spherical, 10.0, &cof), 2.5);
        assert_eq!(value(Model::Cubic, 12.0, &cof), 2.5);
        // 95 % of the sill at the effective range for the exponential model.
        let e = value(Model::Exponential, 10.0, &cof);
        assert!((e - (0.5 + 2.0 * (1.0 - (-3.0f64).exp()))).abs() < 1e-15);
        assert_eq!(value(Model::Matern, 0.0, &[10.0, 2.0, 1.5, 0.5]), 0.5);
    }
}
