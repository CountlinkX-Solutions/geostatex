//! Ordinary cokriging and the LMC fit (R `gstat`).

use crate::kriging::points;
use crate::variogram::f64_binary;
use geostatex_core::cokriging::{self, Cokriging, Kind, Lmc, Sample, Structure};
use rustler::{Binary, Env};

fn kind(name: &str) -> Result<Kind, String> {
    Ok(match name {
        "nugget" => Kind::Nugget,
        "spherical" => Kind::Spherical,
        "exponential" => Kind::Exponential,
        "gaussian" => Kind::Gaussian,
        other => return Err(format!("unknown structure model {other:?}")),
    })
}

fn kind_name(kind: Kind) -> &'static str {
    match kind {
        Kind::Nugget => "nugget",
        Kind::Spherical => "spherical",
        Kind::Exponential => "exponential",
        Kind::Gaussian => "gaussian",
    }
}

/// A structure on the wire: its model, range and sill matrix (row-major f64 binary).
type StructureSpec<'a> = (String, f64, Binary<'a>);

fn lmc(n: usize, structures: Vec<StructureSpec>) -> Result<Lmc, String> {
    let structures = structures
        .into_iter()
        .map(|(name, range, sills)| {
            Ok(Structure {
                kind: kind(&name)?,
                range,
                sills: geostatex_core::bytes_to_f64s(sills.as_slice())?,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    Lmc::new(n, structures)
}

/// Each variable's estimates (f64 binaries) and the error covariance matrices (one f64 binary,
/// `n x n` per target) at `targets` (`(f64 binary, dim)`); no targets only validates.
#[rustler::nif(schedule = "DirtyCpu")]
fn cokriging_predict<'a>(
    env: Env<'a>,
    variables: Vec<(Binary, Binary)>,
    structures: Vec<StructureSpec>,
    targets: (Binary, usize),
) -> Result<(Vec<Binary<'a>>, Binary<'a>), String> {
    let lmc = lmc(variables.len(), structures)?;
    let dim = targets.1;
    let (mut ps, mut vs) = (Vec::new(), Vec::new());
    for (p, v) in variables {
        ps.push(points(p.as_slice(), dim)?);
        vs.push(geostatex_core::bytes_to_f64s(v.as_slice())?);
    }
    let t = points(targets.0.as_slice(), dim)?;
    let ck = Cokriging {
        points: ps,
        values: vs,
        lmc,
    };
    let (estimates, covariances) = ck.predict(&t)?;
    Ok((
        estimates.iter().map(|e| f64_binary(env, e)).collect(),
        f64_binary(env, &covariances.concat()),
    ))
}

/// gstat's `fit.lmc` with fixed ranges: `samples` as `(a, b, dist, gamma, np)` per pair, the
/// structures as `(model, range)`; returns them with their fitted sill matrices.
#[rustler::nif(schedule = "DirtyCpu")]
fn cokriging_fit_lmc<'a>(
    env: Env<'a>,
    n: usize,
    samples: Vec<(usize, usize, Binary, Binary, Binary)>,
    structures: Vec<(String, f64)>,
    correct_diagonal: f64,
) -> Result<Vec<(String, f64, Binary<'a>)>, String> {
    let samples = samples
        .into_iter()
        .map(|(a, b, dist, gamma, np)| {
            Ok((
                a,
                b,
                Sample {
                    dist: geostatex_core::bytes_to_f64s(dist.as_slice())?,
                    gamma: geostatex_core::bytes_to_f64s(gamma.as_slice())?,
                    np: geostatex_core::bytes_to_f64s(np.as_slice())?,
                },
            ))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let kinds = structures
        .iter()
        .map(|(name, range)| Ok((kind(name)?, *range)))
        .collect::<Result<Vec<_>, String>>()?;
    let fitted = cokriging::fit_lmc(n, &samples, &kinds, correct_diagonal)?;
    Ok(fitted
        .structures
        .iter()
        .map(|s| {
            (
                kind_name(s.kind).to_string(),
                s.range,
                f64_binary(env, &s.sills),
            )
        })
        .collect())
}
