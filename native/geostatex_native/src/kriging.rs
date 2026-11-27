//! Ordinary and universal kriging NIFs (2D and 3D). The kriging model is kept in Rust memory as
//! a resource.

use crate::variogram::f64_binary;
use geostatex_core::kriging::{self, Drift, Inverse, Kriging, Model, Point};
use rustler::{Binary, Env, ResourceArc};

pub struct KrigingResource(pub(crate) Kriging);

#[rustler::resource_impl]
impl rustler::Resource for KrigingResource {}

pub(crate) fn model(name: &str) -> Result<Model, String> {
    Ok(match name {
        "linear" => Model::Linear,
        "power" => Model::Power,
        "gaussian" => Model::Gaussian,
        "spherical" => Model::Spherical,
        "exponential" => Model::Exponential,
        "hole_effect" => Model::HoleEffect,
        "custom" => Model::Custom,
        other => match geostatex_core::gstools::GsModel::from_name(other) {
            Some(g) => Model::Gstools(g),
            None => return Err(format!("unknown variogram_model {other:?}")),
        },
    })
}

/// Points of dimension `dim` (2 or 3); 2D points get `z = 0`.
pub(crate) fn points(bytes: &[u8], dim: usize) -> Result<Vec<Point>, String> {
    let flat = geostatex_core::bytes_to_f64s(bytes)?;
    match dim {
        2 => Ok(flat
            .as_chunks::<2>()
            .0
            .iter()
            .map(|p| [p[0], p[1], 0.0])
            .collect()),
        3 => Ok(flat.as_chunks::<3>().0.to_vec()),
        _ => Err("points must be 2D or 3D".to_string()),
    }
}

type KrigingTerm<'a> = (
    ResourceArc<KrigingResource>,
    Binary<'a>,
    Binary<'a>,
    Binary<'a>,
);

/// The drift terms on the wire: the dimension of the regional linear drift (0 for none, 2 or 3),
/// wells `[x, y, Q]` (flat), an external grid `(x, y, z)` and extra data columns (specified and
/// functional drifts).
type DriftSpec<'a> = (
    u8,
    Binary<'a>,
    Option<(Binary<'a>, Binary<'a>, Binary<'a>)>,
    Vec<Binary<'a>>,
);

fn drifts(
    spec: &DriftSpec,
    center: Point,
    scaling: [f64; 2],
    angles: [f64; 3],
) -> Result<(Vec<Drift>, Vec<Vec<f64>>), String> {
    let mut out = Vec::new();
    match spec.0 {
        2 => out.push(Drift::RegionalLinear),
        3 => out.push(Drift::RegionalLinear3),
        _ => {}
    }
    let wells = geostatex_core::bytes_to_f64s(spec.1.as_slice())?;
    if !wells.is_empty() {
        let raw: Vec<[f64; 3]> = wells.as_chunks::<3>().0.to_vec();
        let xy: Vec<Point> = raw.iter().map(|w| [w[0], w[1], 0.0]).collect();
        let adjusted = kriging::adjust(&xy, center, scaling, angles);
        out.push(Drift::PointLog(
            adjusted
                .iter()
                .zip(&raw)
                .map(|(a, w)| [a[0], a[1], w[2]])
                .collect(),
        ));
    }
    if let Some((x, y, z)) = &spec.2 {
        out.push(Drift::External {
            x: geostatex_core::bytes_to_f64s(x.as_slice())?,
            y: geostatex_core::bytes_to_f64s(y.as_slice())?,
            z: geostatex_core::bytes_to_f64s(z.as_slice())?,
        });
    }
    let extra = spec
        .3
        .iter()
        .map(|b| geostatex_core::bytes_to_f64s(b.as_slice()))
        .collect::<Result<Vec<_>, _>>()?;
    Ok((out, extra))
}

// `env` counts as an argument; the options are already grouped in tuples.
#[allow(clippy::too_many_arguments)]
#[rustler::nif(schedule = "DirtyCpu")]
fn kriging_new<'a>(
    env: Env<'a>,
    coords: (Binary, usize),
    values: Binary,
    model_name: String,
    params: Binary,
    fit_spec: (usize, bool),
    // Geographic, exact values, the inverse and the mean of simple kriging (`None` otherwise).
    solve_spec: (bool, bool, String, Option<f64>),
    anisotropy: (f64, f64, f64, f64, f64),
    drift_spec: DriftSpec,
) -> Result<KrigingTerm<'a>, String> {
    let (geographic, exact_values, inverse_name, simple_mean) = solve_spec;
    let raw = points(coords.0.as_slice(), coords.1)?;
    let values = geostatex_core::bytes_to_f64s(values.as_slice())?;
    if raw.len() != values.len() || raw.len() < 2 {
        return Err("kriging needs at least 2 points with one value each".to_string());
    }
    let model = model(&model_name)?;
    let (sy, sz, ax, ay, az) = anisotropy;
    let (scaling, angles) = if geographic {
        ([1.0, 1.0], [0.0; 3])
    } else {
        ([sy, sz], [ax, ay, az])
    };
    let center = if geographic {
        [0.0; 3]
    } else {
        let mid = |k: usize| {
            let max = raw.iter().map(|p| p[k]).fold(f64::NEG_INFINITY, f64::max);
            let min = raw.iter().map(|p| p[k]).fold(f64::INFINITY, f64::min);
            (max + min) / 2.0
        };
        [mid(0), mid(1), mid(2)]
    };
    let adjusted = if geographic {
        raw.clone()
    } else {
        kriging::adjust(&raw, center, scaling, angles)
    };
    let (drift_terms, extra) = drifts(&drift_spec, center, scaling, angles)?;
    if simple_mean.is_some() {
        if !matches!(
            model,
            Model::Spherical | Model::Exponential | Model::Gaussian
        ) {
            return Err(
                "simple kriging needs a model with a sill: spherical, exponential or gaussian"
                    .to_string(),
            );
        }
        if geographic || !drift_terms.is_empty() || !extra.is_empty() {
            return Err(
                "simple kriging takes euclidean coordinates and no drift terms".to_string(),
            );
        }
    }
    if extra.iter().any(|c| c.len() != raw.len()) {
        return Err(
            "every specified and functional drift needs one value per data point".to_string(),
        );
    }
    let mut data_drift = kriging::drift_columns(&drift_terms, &adjusted, &raw)?;
    data_drift.extend(extra);
    let (lags, sv) = kriging::lags(&adjusted, &values, fit_spec.0, geographic);
    let explicit = geostatex_core::bytes_to_f64s(params.as_slice())?;
    let params = if explicit.is_empty() {
        kriging::fit(model, &lags, &sv, fit_spec.1)?
    } else {
        explicit
    };
    let inverse = match inverse_name.as_str() {
        "exact" => Inverse::Exact,
        "pinv" => Inverse::Pinv,
        "pinvh" => Inverse::Pinvh,
        other => return Err(format!("unknown pseudo_inv_type {other:?}")),
    };
    let k = Kriging {
        points: adjusted,
        values,
        model,
        params: params.clone(),
        geographic,
        exact_values,
        inverse,
        center,
        scaling,
        angles,
        drifts: drift_terms,
        data_drift,
        gamma_table: None,
        simple_mean,
    };
    Ok((
        ResourceArc::new(KrigingResource(k)),
        f64_binary(env, &params),
        f64_binary(env, &lags),
        f64_binary(env, &sv),
    ))
}

/// The targets as raw and as the model sees them (anisotropy-adjusted unless geographic).
fn targets_of(k: &Kriging, targets: &(Binary, usize)) -> Result<(Vec<Point>, Vec<Point>), String> {
    let raw = points(targets.0.as_slice(), targets.1)?;
    let adjusted = if k.geographic {
        raw.clone()
    } else {
        kriging::adjust(&raw, k.center, k.scaling, k.angles)
    };
    Ok((raw, adjusted))
}

/// A custom variogram's values on the wire: the distances of `kriging_distances` and the
/// user function at each.
type GammaTable<'a> = Option<(Binary<'a>, Binary<'a>)>;

/// The model with a custom variogram's table, or the model itself.
fn with_table<'k>(
    k: &'k Kriging,
    table: GammaTable,
) -> Result<std::borrow::Cow<'k, Kriging>, String> {
    let Some((d, g)) = table else {
        return Ok(std::borrow::Cow::Borrowed(k));
    };
    let d = geostatex_core::bytes_to_f64s(d.as_slice())?;
    let g = geostatex_core::bytes_to_f64s(g.as_slice())?;
    if d.len() != g.len() {
        return Err("the custom variogram needs one value per distance".to_string());
    }
    let mut custom = k.clone();
    custom.gamma_table = Some(d.iter().map(|x| x.to_bits()).zip(g).collect());
    Ok(std::borrow::Cow::Owned(custom))
}

/// The distances a custom variogram must be evaluated at to krige `targets` (none: the
/// statistics), sorted and without repeats.
#[rustler::nif(schedule = "DirtyCpu")]
fn kriging_distances<'a>(
    env: Env<'a>,
    resource: ResourceArc<KrigingResource>,
    targets: (Binary, usize),
    mask: Binary,
) -> Result<Binary<'a>, String> {
    let k = &resource.0;
    let (_, adjusted) = targets_of(k, &targets)?;
    let mask: Vec<bool> = mask.as_slice().iter().map(|&b| b != 0).collect();
    Ok(f64_binary(env, &k.distances(&adjusted, &mask)?))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn kriging_execute<'a>(
    env: Env<'a>,
    resource: ResourceArc<KrigingResource>,
    // The targets, their dimension and an optional block: `[dx, dy, weight]` offsets.
    targets: (Binary, usize, Option<Binary>),
    extra: Vec<Binary>,
    mask: Binary,
    n_closest: usize,
    table: GammaTable,
) -> Result<(Binary<'a>, Binary<'a>), String> {
    let (bin, dim, block) = targets;
    let k = with_table(&resource.0, table)?;
    let (raw, adjusted) = targets_of(&k, &(bin, dim))?;
    let mask: Vec<bool> = mask.as_slice().iter().map(|&b| b != 0).collect();
    if mask.len() != adjusted.len() {
        return Err("the mask must have one entry per target".to_string());
    }
    if let Some(block) = block {
        let flat = geostatex_core::bytes_to_f64s(block.as_slice())?;
        let triples = flat.as_chunks::<3>().0;
        let offsets: Vec<Point> = triples.iter().map(|o| [o[0], o[1], 0.0]).collect();
        // Offsets turn with the anisotropy (its linear part: the adjustment about the origin).
        let offsets = if k.geographic {
            offsets
        } else {
            kriging::adjust(&offsets, [0.0; 3], k.scaling, k.angles)
        };
        let block: Vec<[f64; 3]> = offsets
            .iter()
            .zip(triples)
            .map(|(o, t)| [o[0], o[1], t[2]])
            .collect();
        let (z, s) = k.execute_block(&adjusted, &mask, &block)?;
        return Ok((f64_binary(env, &z), f64_binary(env, &s)));
    }
    let mut target_drift = kriging::drift_columns(&k.drifts, &adjusted, &raw)?;
    for b in &extra {
        let col = geostatex_core::bytes_to_f64s(b.as_slice())?;
        if col.len() != adjusted.len() {
            return Err(
                "every specified and functional drift needs one value per target".to_string(),
            );
        }
        target_drift.push(col);
    }
    let n_closest = if n_closest == 0 {
        None
    } else {
        Some(n_closest)
    };
    let (z, s) = k.execute(&adjusted, &target_drift, &mask, n_closest)?;
    Ok((f64_binary(env, &z), f64_binary(env, &s)))
}

#[rustler::nif]
fn kriging_objective(
    model_name: String,
    params: Binary,
    lags: Binary,
    sv: Binary,
    weight: bool,
) -> Result<f64, String> {
    let params = geostatex_core::bytes_to_f64s(params.as_slice())?;
    let lags = geostatex_core::bytes_to_f64s(lags.as_slice())?;
    let sv = geostatex_core::bytes_to_f64s(sv.as_slice())?;
    Ok(kriging::objective(
        model(&model_name)?,
        &params,
        &lags,
        &sv,
        weight,
    ))
}

/// PyKrige's model statistics: the residuals and kriging standard deviations (f64 binaries).
#[rustler::nif(schedule = "DirtyCpu")]
fn kriging_statistics<'a>(
    env: Env<'a>,
    resource: ResourceArc<KrigingResource>,
    table: GammaTable,
) -> Result<(Binary<'a>, Binary<'a>), String> {
    let (delta, sigma) = with_table(&resource.0, table)?.statistics()?;
    Ok((f64_binary(env, &delta), f64_binary(env, &sigma)))
}
