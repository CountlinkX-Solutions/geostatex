//! Spatial random fields (`gstools.SRF`, `gstools.CondSRF`).

use crate::kriging::{model, points};
use crate::variogram::f64_binary;
use geostatex_core::kriging::{self, Model};
use geostatex_core::simulation::{self, Center, Spec};
use rustler::{Binary, Env};

/// The model on the wire: its name, `[psill, range, nugget]` and the anisotropy as kriging's
/// (y and z scaling, x, y and z angles in degrees), applied about the origin.
type ModelSpec<'a> = (String, Binary<'a>, (f64, f64, f64, f64, f64));

/// One field at `positions` (`(f64 binary, dim)`) from `(mode_no, seed, mean)`; conditioned on
/// the kriging estimate and variance at the positions when given (the mean is then unused).
#[rustler::nif(schedule = "DirtyCpu")]
fn simulation_field<'a>(
    env: Env<'a>,
    model_spec: ModelSpec,
    generator: (usize, u32, f64),
    positions: (Binary, usize),
    condition: Option<(Binary, Binary)>,
) -> Result<Binary<'a>, String> {
    let (name, params, (sy, sz, ax, ay, az)) = model_spec;
    let model = model(&name)?;
    if !matches!(
        model,
        Model::Spherical | Model::Exponential | Model::Gaussian
    ) {
        return Err("simulation needs a spherical, exponential or gaussian model".to_string());
    }
    let params = geostatex_core::bytes_to_f64s(params.as_slice())?;
    let [psill, range, nugget] = params[..] else {
        return Err("the model needs [psill, range, nugget]".to_string());
    };
    let (mode_no, seed, mean) = generator;
    let raw = points(positions.0.as_slice(), positions.1)?;
    let iso = kriging::adjust(&raw, [0.0; 3], [sy, sz], [ax, ay, az]);
    let spec = Spec {
        model,
        psill,
        range,
        nugget,
        dim: positions.1,
        mode_no,
    };
    let out = match condition {
        None => simulation::field(&spec, &iso, seed, Center::Mean(mean))?,
        Some((estimate, variance)) => {
            let estimate = geostatex_core::bytes_to_f64s(estimate.as_slice())?;
            let variance = geostatex_core::bytes_to_f64s(variance.as_slice())?;
            simulation::field(
                &spec,
                &iso,
                seed,
                Center::Kriging {
                    estimate: &estimate,
                    variance: &variance,
                },
            )?
        }
    };
    Ok(f64_binary(env, &out))
}
