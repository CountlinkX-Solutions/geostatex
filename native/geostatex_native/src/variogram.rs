//! Experimental variogram NIF.

use geostatex_core::variogram::{self, Bins, Estimator, MaxLag, Metric, Options, Rule};
use rustler::{Binary, Env, OwnedBinary};

fn metric(name: &str) -> Result<Metric, String> {
    match name {
        "euclidean" => Ok(Metric::Euclidean),
        "cityblock" => Ok(Metric::Cityblock),
        "chebyshev" => Ok(Metric::Chebyshev),
        "sqeuclidean" => Ok(Metric::SqEuclidean),
        "seuclidean" => Ok(Metric::SEuclidean),
        "canberra" => Ok(Metric::Canberra),
        "braycurtis" => Ok(Metric::Braycurtis),
        "cosine" => Ok(Metric::Cosine),
        "mahalanobis" => Ok(Metric::Mahalanobis),
        "minkowski" => Ok(Metric::Minkowski),
        other => Err(format!("unknown dist_func {other:?}")),
    }
}

fn maxlag(kind: &str, value: f64) -> Result<MaxLag, String> {
    match kind {
        "unset" => Ok(MaxLag::Unset),
        "value" => Ok(MaxLag::Value(value)),
        "mean" => Ok(MaxLag::Mean),
        "median" => Ok(MaxLag::Median),
        other => Err(format!("unknown maxlag {other:?}")),
    }
}

fn bins(kind: &str, edges: &[u8]) -> Result<Bins, String> {
    Ok(match kind {
        "even" => Bins::Even,
        "uniform" => Bins::Uniform,
        "sturges" => Bins::Rule(Rule::Sturges),
        "scott" => Bins::Rule(Rule::Scott),
        "sqrt" => Bins::Rule(Rule::Sqrt),
        "fd" => Bins::Rule(Rule::Fd),
        "doane" => Bins::Rule(Rule::Doane),
        "rice" => Bins::Rule(Rule::Rice),
        "edges" => Bins::Edges(geostatex_core::bytes_to_f64s(edges)?),
        other => return Err(format!("unknown bin_func {other:?}")),
    })
}

fn estimator(kind: &str, param: f64) -> Result<Estimator, String> {
    Ok(match kind {
        "matheron" => Estimator::Matheron,
        "cressie" => Estimator::Cressie,
        "dowd" => Estimator::Dowd,
        "genton" => Estimator::Genton,
        "minmax" => Estimator::Minmax,
        "percentile" => Estimator::Percentile(param),
        "entropy" => Estimator::Entropy(param as usize),
        other => return Err(format!("unknown estimator {other:?}")),
    })
}

pub(crate) fn f64_binary<'a>(env: Env<'a>, values: &[f64]) -> Binary<'a> {
    let mut owned = OwnedBinary::new(values.len() * 8);
    let (slots, _) = owned.as_mut_slice().as_chunks_mut::<8>();
    for (slot, value) in slots.iter_mut().zip(values) {
        *slot = value.to_ne_bytes();
    }
    owned.release(env)
}

/// Bins, semivariances, counts, maxlag, bandwidth and the entropy fit sigma (empty unless asked).
type ExperimentalTerm<'a> = (
    Binary<'a>,
    Binary<'a>,
    Vec<u64>,
    Option<f64>,
    Option<f64>,
    Binary<'a>,
);

/// A direction on the wire: azimuth, tolerance, bandwidth (`"value"` or `"percentile"` and its
/// number) and search area.
type DirectionSpec = (f64, f64, (String, f64), String);

/// Which pairs enter (a direction, a sample of `k` points per side with its seed) and how their
/// values combine (a co-variable: `"skgstat"` or `"classical"` and its values).
type PairsSpec<'a> = (
    Option<DirectionSpec>,
    Option<(String, Binary<'a>)>,
    Option<(usize, u64)>,
);

fn direction(spec: Option<DirectionSpec>) -> Result<Option<variogram::Direction>, String> {
    let Some((azimuth, tolerance, (kind, width), area)) = spec else {
        return Ok(None);
    };
    Ok(Some(variogram::Direction {
        azimuth,
        tolerance,
        bandwidth: match kind.as_str() {
            "value" => variogram::Bandwidth::Value(width),
            "percentile" => variogram::Bandwidth::Percentile(width),
            other => return Err(format!("unknown bandwidth kind {other:?}")),
        },
        area: match area.as_str() {
            "triangle" => variogram::SearchArea::Triangle,
            "compass" => variogram::SearchArea::Compass,
            other => return Err(format!("unknown directional model {other:?}")),
        },
    }))
}

/// A cross-variogram on the wire: `"skgstat"` or `"classical"` and the co-variable.
fn cross(spec: Option<(String, Binary)>) -> Result<Option<variogram::Cross>, String> {
    let Some((kind, co)) = spec else {
        return Ok(None);
    };
    let co = geostatex_core::bytes_to_f64s(co.as_slice())?;
    Ok(Some(match kind.as_str() {
        "skgstat" => variogram::Cross::Skgstat(co),
        "classical" => variogram::Cross::Classical(co),
        other => return Err(format!("unknown cross {other:?}")),
    }))
}

#[allow(clippy::too_many_arguments)]
#[rustler::nif(schedule = "DirtyCpu")]
fn variogram_experimental<'a>(
    env: Env<'a>,
    coords: Binary,
    dim: usize,
    values: Binary,
    dist_func: String,
    maxlag_spec: (String, f64),
    bins_spec: (String, Binary, usize),
    // The estimator, its parameter and whether to compute the entropy fit sigma.
    estimator_spec: (String, f64, bool),
    pairs_spec: PairsSpec<'_>,
) -> Result<ExperimentalTerm<'a>, String> {
    let (direction_spec, cross_spec, sampling_spec) = pairs_spec;
    let coords = geostatex_core::bytes_to_f64s(coords.as_slice())?;
    let values = geostatex_core::bytes_to_f64s(values.as_slice())?;
    let opts = Options {
        metric: metric(&dist_func)?,
        maxlag: maxlag(&maxlag_spec.0, maxlag_spec.1)?,
        bins: bins(&bins_spec.0, bins_spec.1.as_slice())?,
        n_lags: bins_spec.2,
        estimator: estimator(&estimator_spec.0, estimator_spec.1)?,
        direction: direction(direction_spec)?,
        cross: cross(cross_spec)?,
        sampling: sampling_spec.map(|(k, seed)| variogram::Sampling { k, seed }),
        entropy_sigma: estimator_spec.2,
    };
    let v = variogram::experimental(&coords, dim, &values, &opts)?;
    Ok((
        f64_binary(env, &v.bins),
        f64_binary(env, &v.experimental),
        v.bin_count.iter().map(|&c| c as u64).collect(),
        v.maxlag,
        v.bandwidth,
        f64_binary(env, &v.entropy_sigma.unwrap_or_default()),
    ))
}

fn model(name: &str) -> Result<geostatex_core::models::Model, String> {
    use geostatex_core::models::Model;
    Ok(match name {
        "spherical" => Model::Spherical,
        "exponential" => Model::Exponential,
        "gaussian" => Model::Gaussian,
        "cubic" => Model::Cubic,
        "stable" => Model::Stable,
        "matern" => Model::Matern,
        other => match geostatex_core::gstools::GsModel::from_name(other) {
            Some(g) if g.shapes() <= 1 => Model::Gstools(g),
            Some(_) => {
                return Err(format!(
                    "{other} has two shapes: it is available in kriging only"
                ))
            }
            None => return Err(format!("unknown model {other:?}")),
        },
    })
}

/// The components of a summed model name (`"spherical+gaussian"`), or `None` for one model.
fn components(name: &str) -> Result<Option<Vec<geostatex_core::models::Model>>, String> {
    if !name.contains('+') {
        return Ok(None);
    }
    name.split('+')
        .map(model)
        .collect::<Result<_, _>>()
        .map(Some)
}

fn method(kind: &str, manual: &[u8]) -> Result<geostatex_core::fit::Method, String> {
    use geostatex_core::fit::Method;
    Ok(match kind {
        "trf" => Method::Trf,
        "lm" => Method::Lm,
        "ml" => Method::Ml,
        "manual" => {
            let v = geostatex_core::bytes_to_f64s(manual)?;
            let p: [f64; 4] = v.try_into().map_err(|_| "manual fit needs 4 values")?;
            Method::Manual(p)
        }
        other => return Err(format!("unknown fit_method {other:?}")),
    })
}

fn sigma(kind: &str, values: &[u8]) -> Result<geostatex_core::fit::Sigma, String> {
    use geostatex_core::fit::Sigma;
    Ok(match kind {
        "unit" => Sigma::Unit,
        "linear" => Sigma::Linear,
        "exp" => Sigma::Exp,
        "sqrt" => Sigma::Sqrt,
        "sq" => Sigma::Sq,
        "values" => Sigma::Values(geostatex_core::bytes_to_f64s(values)?),
        other => return Err(format!("unknown fit_sigma {other:?}")),
    })
}

type FitTerm<'a> = (Binary<'a>, Binary<'a>, Binary<'a>, Binary<'a>);

#[rustler::nif(schedule = "DirtyCpu")]
fn variogram_fit<'a>(
    env: Env<'a>,
    bins: Binary,
    experimental: Binary,
    model_name: String,
    // The method, the manual parameters, scikit-gstat's fit_bounds and fit_p0.
    method_spec: (String, Binary, Option<(Binary, Binary)>, Option<Binary>),
    sigma_spec: (String, Binary),
    use_nugget: bool,
) -> Result<FitTerm<'a>, String> {
    let bins = geostatex_core::bytes_to_f64s(bins.as_slice())?;
    let experimental = geostatex_core::bytes_to_f64s(experimental.as_slice())?;
    // A harmonized variogram has no parameters to fit: the method and sigma do not apply.
    let f = if model_name == "harmonize" {
        geostatex_core::fit::harmonize(&bins, &experimental, use_nugget)?
    } else if let Some(models) = components(&model_name)? {
        geostatex_core::fit::fit_sum(
            &models,
            &bins,
            &experimental,
            method(&method_spec.0, method_spec.1.as_slice())?,
            &sigma(&sigma_spec.0, sigma_spec.1.as_slice())?,
            use_nugget,
        )?
    } else {
        let floats = |b: &Binary| geostatex_core::bytes_to_f64s(b.as_slice());
        let opts = geostatex_core::fit::FitOptions {
            model: model(&model_name)?,
            method: method(&method_spec.0, method_spec.1.as_slice())?,
            sigma: sigma(&sigma_spec.0, sigma_spec.1.as_slice())?,
            use_nugget,
            bounds: match &method_spec.2 {
                Some((l, u)) => Some((floats(l)?, floats(u)?)),
                None => None,
            },
            p0: method_spec.3.as_ref().map(floats).transpose()?,
        };
        geostatex_core::fit::fit(&bins, &experimental, &opts)?
    };
    let diagnostics = [f.rmse, f.mse, f.mae, f.nrmse, f.nrmse_r, f.r, f.nse];
    Ok((
        f64_binary(env, &f.cof),
        f64_binary(env, &f.parameters),
        f64_binary(env, &f.model_at_bins),
        f64_binary(env, &diagnostics),
    ))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn variogram_model_values<'a>(
    env: Env<'a>,
    model_name: String,
    cof: Binary,
    h: Binary,
) -> Result<Binary<'a>, String> {
    let cof = geostatex_core::bytes_to_f64s(cof.as_slice())?;
    let h = geostatex_core::bytes_to_f64s(h.as_slice())?;
    let values: Vec<f64> = match components(&model_name)? {
        Some(models) => {
            let size = geostatex_core::fit::sum_size(&models);
            if cof.len() != size && cof.len() != size + 1 {
                return Err(format!(
                    "the summed model needs {size} coefficients and optionally a nugget"
                ));
            }
            h.iter()
                .map(|&x| geostatex_core::fit::sum_value(&models, x, &cof))
                .collect()
        }
        None => {
            let m = model(&model_name)?;
            let cof = geostatex_core::fit::full_cof(m, &cof);
            h.iter()
                .map(|&x| geostatex_core::models::value(m, x, &cof))
                .collect()
        }
    };
    Ok(f64_binary(env, &values))
}

type SpaceTimeTerm<'a> = (Binary<'a>, Binary<'a>, Binary<'a>, Vec<u64>, Option<f64>);

/// Space-time experimental variogram: `x_spec` and `t_spec` are the lag binning of each axis
/// (kind, explicit edges, lag count; 0 time lags for one per step).
#[allow(clippy::too_many_arguments)]
#[rustler::nif(schedule = "DirtyCpu")]
fn spacetime_experimental<'a>(
    env: Env<'a>,
    coords: (Binary, usize),
    values: (Binary, usize),
    dist_func: String,
    x_spec: (String, Binary, usize),
    maxlag_spec: (String, f64),
    t_spec: (String, Binary, usize),
    estimator_spec: (String, f64),
) -> Result<SpaceTimeTerm<'a>, String> {
    let flat = geostatex_core::bytes_to_f64s(coords.0.as_slice())?;
    let vals = geostatex_core::bytes_to_f64s(values.0.as_slice())?;
    let opts = geostatex_core::spacetime::Options {
        metric: metric(&dist_func)?,
        x_bins: bins(&x_spec.0, x_spec.1.as_slice())?,
        x_lags: x_spec.2,
        maxlag: maxlag(&maxlag_spec.0, maxlag_spec.1)?,
        t_bins: bins(&t_spec.0, t_spec.1.as_slice())?,
        t_lags: (t_spec.2 > 0).then_some(t_spec.2),
        estimator: estimator(&estimator_spec.0, estimator_spec.1)?,
    };
    let st = geostatex_core::spacetime::experimental(&flat, coords.1, &vals, values.1, &opts)?;
    Ok((
        f64_binary(env, &st.xbins),
        f64_binary(env, &st.tbins),
        f64_binary(env, &st.experimental),
        st.counts.iter().map(|&c| c as u64).collect(),
        st.maxlag,
    ))
}
