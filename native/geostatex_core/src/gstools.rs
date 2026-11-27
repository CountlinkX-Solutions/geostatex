//! GSTools' covariance models that scikit-gstat and PyKrige do not have (GSTools 1.7.0
//! `covmodel`), as correlation functions of the normalised distance `h = r / len_scale`
//! (`rescale = 1`), and the special functions they need.

use std::f64::consts::PI;

const EULER: f64 = 0.577_215_664_901_532_9;

/// A GSTools model (truncated power laws with `len_low = 0`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GsModel {
    Integral,
    Rational,
    Circular,
    SuperSpherical,
    JBessel,
    TplGaussian,
    TplExponential,
    TplStable,
    TplSimple,
}

impl GsModel {
    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "integral" => GsModel::Integral,
            "rational" => GsModel::Rational,
            "circular" => GsModel::Circular,
            "superspherical" => GsModel::SuperSpherical,
            "jbessel" => GsModel::JBessel,
            "tpl_gaussian" => GsModel::TplGaussian,
            "tpl_exponential" => GsModel::TplExponential,
            "tpl_stable" => GsModel::TplStable,
            "tpl_simple" => GsModel::TplSimple,
            _ => return None,
        })
    }

    /// The number of shape parameters (`nu`, `alpha` or `hurst`; `[hurst, alpha]` for
    /// `TplStable`).
    pub fn shapes(self) -> usize {
        match self {
            GsModel::Circular => 0,
            GsModel::TplStable => 2,
            _ => 1,
        }
    }

    /// GSTools' default bounds of the single shape parameter in 2D (for fits).
    pub fn shape_bounds(self) -> (f64, f64) {
        match self {
            GsModel::Integral => (0.0, 50.0),
            GsModel::Rational => (0.5, 50.0),
            GsModel::SuperSpherical => (0.5, 50.0),
            GsModel::JBessel => (0.0, 50.0),
            GsModel::TplGaussian | GsModel::TplExponential => (0.1, 1.0),
            GsModel::TplSimple => (1.5, 50.0),
            GsModel::Circular | GsModel::TplStable => (f64::NAN, f64::NAN),
        }
    }
}

/// The correlation at the normalised distance `h` (`shape` as `GsModel::shapes`).
pub fn cor(model: GsModel, h: f64, shape: &[f64]) -> f64 {
    let h = h.abs();
    match model {
        GsModel::Integral => 0.5 * shape[0] * exp_int(1.0 + 0.5 * shape[0], h * h),
        GsModel::Rational => (1.0 + h * h / shape[0]).powf(-shape[0]),
        GsModel::Circular => {
            if h < 1.0 {
                2.0 / PI * (h.acos() - h * (1.0 - h * h).sqrt())
            } else {
                0.0
            }
        }
        // `1 - h 2F1(1/2, -nu; 3/2; h^2) / 2F1(1/2, -nu; 3/2; 1)` is `1 - I_{h^2}(1/2, nu + 1)`.
        GsModel::SuperSpherical => {
            if h < 1.0 {
                1.0 - inc_beta(0.5, shape[0] + 1.0, h * h)
            } else {
                0.0
            }
        }
        GsModel::JBessel => bessel_j_normalised(shape[0], h),
        GsModel::TplGaussian => tpl_stable(h, shape[0], 2.0),
        GsModel::TplExponential => tpl_stable(h, shape[0], 1.0),
        GsModel::TplStable => tpl_stable(h, shape[0], shape[1]),
        GsModel::TplSimple => {
            if h < 1.0 {
                (1.0 - h).powf(shape[0])
            } else {
                0.0
            }
        }
    }
}

/// GSTools' `tplstable_cor`: `2H/alpha E_{1 + 2H/alpha}(h^alpha)`, and 1 where
/// `np.isclose(h, 0)` (`h <= 1e-8`), as GSTools sets it.
fn tpl_stable(h: f64, hurst: f64, alpha: f64) -> f64 {
    if h <= 1e-8 {
        return 1.0;
    }
    let s = 2.0 * hurst / alpha;
    s * exp_int(1.0 + s, h.powf(alpha))
}

/// `b0 + a1 / (b1 + a2 / (b2 + ...))` by the modified Lentz algorithm (NIST DLMF 3.10(iii)),
/// with `term(n) = (a_n, b_n)` for `n >= 1`, until a term changes the value by less than one
/// unit of rounding.
fn continued_fraction(b0: f64, term: impl Fn(u32) -> (f64, f64)) -> f64 {
    // A vanishing denominator is replaced by the smallest magnitude worth dividing by.
    let nonzero = |v: f64| if v.abs() < 1e-300 { 1e-300 } else { v };
    let mut value = nonzero(b0);
    let (mut ratio, mut inverse) = (value, 0.0);
    for n in 1..=20_000 {
        let (a, b) = term(n);
        inverse = 1.0 / nonzero(b + a * inverse);
        ratio = nonzero(b + a / ratio);
        let step = ratio * inverse;
        value *= step;
        if (step - 1.0).abs() <= f64::EPSILON {
            break;
        }
    }
    value
}

/// The generalised exponential integral `E_p(x) = int_1^inf exp(-x t) / t^p dt` for `x >= 0`.
/// As GSTools, `x <= 1e-20` gives the limit at 0 (`1 / (p - 1)` for `p > 1`). The continued
/// fraction DLMF 8.19.17 for `x > 1`, the power series DLMF 8.19.8 (integer orders) and 8.19.10
/// otherwise.
pub fn exp_int(p: f64, x: f64) -> f64 {
    if x <= 1e-20 {
        return if p > 1.0 {
            1.0 / (p - 1.0)
        } else {
            f64::INFINITY
        };
    }
    if x > 1.0 {
        // exp(-x) / (x + p / (1 + 1 / (x + (p + 1) / (1 + 2 / (x + ...))))).
        let fraction = continued_fraction(0.0, |n| {
            let half = (n / 2) as f64;
            match (n, n % 2) {
                (1, _) => (1.0, x),
                (_, 0) => (p + half - 1.0, 1.0),
                _ => (half, x),
            }
        });
        return (-x).exp() * fraction;
    }
    let order = p.round();
    let integer = ((p - order).abs() < 1e-9 && order >= 1.0).then_some(order as u32);
    // sum_k (-x)^k / (k! (1 - p + k)), without the term an integer order makes singular.
    let singular = integer.map(|n| n - 1);
    let (mut sum, mut power, mut at_singular) = (0.0, 1.0, 0.0);
    for k in 0..60_u32 {
        if k > 0 {
            power *= -x / k as f64;
        }
        if Some(k) == singular {
            at_singular = power;
        } else {
            sum += power / (1.0 - p + k as f64);
        }
    }
    match integer {
        // (-x)^(n-1) / (n-1)! (psi(n) - ln x), with psi(n) = -gamma + sum_{k<n} 1/k.
        Some(n) => {
            let digamma = (1..n).map(|k| 1.0 / k as f64).sum::<f64>() - EULER;
            at_singular * (digamma - x.ln()) - sum
        }
        None => libm::tgamma(1.0 - p) * x.powf(p - 1.0) - sum,
    }
}

/// The regularised incomplete beta function `I_x(a, b)`: the continued fraction DLMF 8.17.22,
/// on the side of the symmetry `I_x(a, b) = 1 - I_{1-x}(b, a)` (DLMF 8.17.4) where it converges
/// fast.
pub fn inc_beta(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    if x > (a + 1.0) / (a + b + 2.0) {
        return 1.0 - inc_beta(b, a, 1.0 - x);
    }
    // x^a (1 - x)^b / (a B(a, b)) * 1 / (1 + d_1 / (1 + d_2 / (1 + ...))).
    let log_front = a * x.ln() + b * (1.0 - x).ln() - a.ln() + libm::lgamma(a + b)
        - libm::lgamma(a)
        - libm::lgamma(b);
    let fraction = continued_fraction(0.0, |n| {
        if n == 1 {
            return (1.0, 1.0);
        }
        let m = ((n - 1) / 2) as f64;
        let d = if (n - 1) % 2 == 0 {
            m * (b - m) * x / ((a + 2.0 * m - 1.0) * (a + 2.0 * m))
        } else {
            -(a + m) * (a + b + m) * x / ((a + 2.0 * m) * (a + 2.0 * m + 1.0))
        };
        (d, 1.0)
    });
    log_front.exp() * fraction
}

/// `Gamma(nu + 1) J_nu(x) / (x / 2)^nu` for `nu >= 0`, `x >= 0` (1 at 0): the series
/// `0F1(; nu + 1; -x^2 / 4)` up to `x = 12`, Hankel's asymptotic expansion beyond (each about
/// 1e-12 there).
pub fn bessel_j_normalised(nu: f64, x: f64) -> f64 {
    if x == 0.0 {
        return 1.0;
    }
    if x <= 12.0 {
        let z = -x * x / 4.0;
        let (mut sum, mut term) = (1.0, 1.0);
        for k in 1..1000 {
            let k = k as f64;
            term *= z / (k * (nu + k));
            sum += term;
            if k > x && term.abs() < 1e-17 * sum.abs().max(1e-300) {
                break;
            }
        }
        return sum;
    }
    let mu = 4.0 * nu * nu;
    let (mut p, mut q, mut term) = (1.0, 0.0, 1.0);
    let mut previous = f64::INFINITY;
    for k in 1..200 {
        let next = term * (mu - (2.0 * k as f64 - 1.0).powi(2)) / (k as f64 * 8.0 * x);
        if next.abs() >= previous.min(term.abs()) && k > 2 {
            break;
        }
        previous = term.abs();
        term = next;
        match k % 4 {
            1 => q += term,
            2 => p -= term,
            3 => q -= term,
            _ => p += term,
        }
        if term.abs() < 1e-17 {
            break;
        }
    }
    let chi = x - (nu / 2.0 + 0.25) * PI;
    let j = (2.0 / (PI * x)).sqrt() * (p * chi.cos() - q * chi.sin());
    (libm::lgamma(nu + 1.0) - nu * (x / 2.0).ln()).exp() * j
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * b.abs().max(1e-300) || (a - b).abs() <= 1e-300
    }

    #[test]
    fn a_continued_fraction_by_lentz() {
        // 1 + 1/(1 + 1/(1 + ...)) is the golden ratio.
        let golden = continued_fraction(1.0, |_| (1.0, 1.0));
        assert!(close(golden, 1.618_033_988_749_895, 1e-15), "{golden}");
        // Lambert: tan x = x/(1 - x^2/(3 - x^2/(5 - ...))), here at x = 1 (a zero leading term).
        let tan = continued_fraction(0.0, |n| {
            (if n == 1 { 1.0 } else { -1.0 }, 2.0 * n as f64 - 1.0)
        });
        assert!(close(tan, 1.0_f64.tan(), 1e-15), "{tan}");
    }

    #[test]
    fn the_exponential_integral_at_integer_and_large_orders() {
        // scipy.special.expn(n, x); the order 30 has its logarithmic term at k = 29.
        for (s, x, want) in [
            (1.0, 0.5, 0.5597735947761608),
            (1.0, 2.0, 0.048900510708061146),
            (3.0, 1.0, 0.1096919671977602),
            (30.0, 0.7, 0.01670634835596143),
        ] {
            let got = exp_int(s, x);
            assert!(close(got, want, 1e-12), "E_{s}({x}) = {got} vs {want}");
        }
    }

    // Reference values from scipy 1.16 (gstools.tools.special.exp_int, betainc, jv, gamma).

    #[test]
    fn the_generalised_exponential_integral() {
        for (s, x, want) in [
            (1.5, 0.5, 0.41768182857857156),
            (1.25, 0.01, 2.4632616773587537),
            (2.0, 0.3, 0.4691152251789638),
            (2.0, 4.0, 0.0031982292493385545),
            (2.8, 5.0, 0.0008993820532290376),
            (4.65, 1.0, 0.0752635439380942),
            (1.75, 16.0, 6.372242127083198e-09),
        ] {
            let got = exp_int(s, x);
            assert!(close(got, want, 1e-12), "E_{s}({x}) = {got} vs {want}");
        }
        assert_eq!(exp_int(1.5, 0.0), 2.0);
    }

    #[test]
    fn the_regularised_incomplete_beta() {
        for (a, b, x, want) in [
            (0.5, 2.0, 0.3, 0.7394254526319747),
            (0.5, 1.5, 0.81, 0.9626139265315015),
            (0.5, 4.7, 0.05, 0.5011277750629577),
        ] {
            let got = inc_beta(a, b, x);
            assert!(close(got, want, 1e-13), "I_{x}({a}, {b}) = {got} vs {want}");
        }
        assert_eq!(
            (inc_beta(0.5, 2.0, 0.0), inc_beta(0.5, 2.0, 1.0)),
            (0.0, 1.0)
        );
    }

    #[test]
    fn the_normalised_bessel_function() {
        // Gamma(nu + 1) J_nu(x) / (x / 2)^nu.
        for (nu, x, want) in [
            (1.0, 3.0, 0.2260393056839575),
            (0.5, 0.2, 0.9933466539753072),
            (2.5, 7.0, -0.0411019196307556),
            (1.0, 18.0, -0.020888320609785505),
            (2.5, 30.0, 0.0005385072446428485),
            (0.5, 60.0, -0.005080177018370279),
        ] {
            let got = bessel_j_normalised(nu, x);
            assert!(
                (got - want).abs() < 1e-12,
                "nu {nu}, x {x}: {got} vs {want}"
            );
        }
        assert_eq!(bessel_j_normalised(1.0, 0.0), 1.0);
    }

    #[test]
    fn variograms_as_gstools() {
        // nugget 0.2 + var 1.3 (1 - cor(r / 10)) at r = 0.05, 0.5, 4.5169..., 17.9067..., 40.
        let r = [0.05, 0.5, 4.516949152542373, 17.906779661016948, 40.0];
        let cases: [(GsModel, &[f64], [f64; 5]); 9] = [
            (
                GsModel::Integral,
                &[0.5],
                [
                    0.3126342265367122,
                    0.5551322028880357,
                    1.185947415304781,
                    1.4968977945304927,
                    1.4999999978716694,
                ],
            ),
            (
                GsModel::Rational,
                &[0.5],
                [
                    0.20003249878130097,
                    0.20323786306007713,
                    0.4044472405146285,
                    1.0225314874862599,
                    1.2736990472575929,
                ],
            ),
            (
                GsModel::Circular,
                &[],
                [
                    0.2082760225570784,
                    0.2827260738938831,
                    0.9213857846136708,
                    1.5,
                    1.5,
                ],
            ),
            (
                GsModel::SuperSpherical,
                &[0.5],
                [
                    0.2082760225570784,
                    0.2827260738938833,
                    0.9213857846136708,
                    1.5,
                    1.5,
                ],
            ),
            (
                GsModel::JBessel,
                &[0.5],
                [
                    0.2000054166598944,
                    0.20054159896236207,
                    0.2437573503313027,
                    0.7914972552225912,
                    1.745960810975075,
                ],
            ),
            (
                GsModel::TplGaussian,
                &[0.25],
                [
                    0.3126342265367122,
                    0.5551322028880357,
                    1.185947415304781,
                    1.4968977945304927,
                    1.4999999978716694,
                ],
            ),
            (
                GsModel::TplExponential,
                &[0.25],
                [
                    0.35643624910584754,
                    0.6507688466782684,
                    1.201915839018949,
                    1.4632652036873717,
                    1.4977464498343946,
                ],
            ),
            (
                GsModel::TplStable,
                &[0.5, 1.5],
                [
                    0.21649390228232163,
                    0.3451026597029517,
                    1.0120910104765217,
                    1.4789346975564286,
                    1.499969456079544,
                ],
            ),
            (
                GsModel::TplSimple,
                &[1.5],
                [
                    0.20973780232465927,
                    0.29627089841609294,
                    0.9721910352713798,
                    1.5,
                    1.5,
                ],
            ),
        ];
        for (model, shape, want) in cases {
            for (d, w) in r.iter().zip(want) {
                let got = 0.2 + 1.3 * (1.0 - cor(model, d / 10.0, shape));
                assert!(close(got, w, 1e-10), "{model:?} at {d}: {got} vs {w}");
            }
            assert_eq!(cor(model, 0.0, shape), 1.0, "{model:?} at 0");
        }
    }

    #[test]
    fn names_and_shapes() {
        for (name, shapes) in [
            ("integral", 1),
            ("rational", 1),
            ("circular", 0),
            ("superspherical", 1),
            ("jbessel", 1),
            ("tpl_gaussian", 1),
            ("tpl_exponential", 1),
            ("tpl_stable", 2),
            ("tpl_simple", 1),
        ] {
            let model = GsModel::from_name(name).unwrap();
            assert_eq!(model.shapes(), shapes, "{name}");
        }
        assert!(GsModel::from_name("matern").is_none());
    }
}
