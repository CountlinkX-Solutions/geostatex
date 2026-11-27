"""Generate the M10 fitting fixtures (`fixtures/variogram/fit_extra.json`).

Run from the repository root (under a memory cap):

    prlimit --as=3221225472 .venv/bin/python -I test/reference/generate_variogram_fit_extra.py

Sections:

* `entropy`: `fit_sigma="entropy"`, whose sigma is `1 / h` per non-empty lag, `h` the Shannon
  entropy of the lag's value differences over `np.histogram_bin_edges(distance, "scott")`. Each
  case records the sigmas, scikit-gstat's coefficients and fitted curve.
* `ml`: `fit_method="ml"`, SLSQP minimising `-sum(norm.logpdf(pred, y, 1) / sigma)` within the
  trf bounds, i.e. `sum((pred - y)^2 / sigma) / 2` plus a constant. Each case records the sigma
  of every non-empty lag (ones without `fit_sigma`) for that objective.
* `harmonize`: `model="harmonize"`, sklearn's `IsotonicRegression(increasing=True)` of the
  non-empty lags, interpolated linearly and NaN outside them. Each case records the model at the
  bins and on a grid, and scikit-gstat's `cof = [r, s, n]`, where `r` indexes
  `linspace(0, last non-empty bin, 100)`; `range` is that distance (Geostatex reports it instead
  of the index). Explicit bin edges are left out: scikit-gstat's harmonize fails on them
  (TypeError on a list of bins).
* `sum_models`: summed models (`"spherical+gaussian"`) evaluated with given coefficients: each
  component's `(range, sill[, shape])` in order and one nugget at the end (`_build_sum_models`).
* `sums`: summed-model fits (trf within the concatenated component bounds, nugget last), with
  scikit-gstat's coefficients and curve, or the failure of its curve_fit.

Datasets: Meuse topsoil lead and zinc (Pebesma and Bivand 2005, bundled with scikit-gstat), the
pancake sample (seed 42, 150 points) and, for `harmonize`, two seeded clusters 100 apart (the
lags between them are empty).
With 12 lags as in `fit.json` unless a case says otherwise.
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import skgstat as skg  # noqa: E402

from common import dump_json, meta  # noqa: E402

warnings.simplefilter("ignore")

N_LAGS = 12


def datasets():
    out = {}
    for var in ("lead", "zinc"):
        c, z = skg.data.meuse(variable=var).get("sample")
        out[f"meuse_{var}"] = (np.asarray(c, float), np.asarray(z, float).ravel())
    c, z = skg.data.pancake(N=150, seed=42).get("sample")
    out["pancake"] = (np.asarray(c, float), np.asarray(z, float).ravel())
    rng = np.random.default_rng(5)
    c = np.vstack((rng.uniform(0, 10, (30, 2)), rng.uniform(0, 10, (30, 2)) + [100, 0]))
    out["clusters"] = (c, 0.02 * c[:, 0] + rng.normal(0, 1, 60))
    return out


def fitted(name, v, params, **extra):
    return {"dataset": name, "params": params, "bins": v.bins, "experimental": v.experimental,
            "cof": list(v.cof), "model_at_bins": v.transform(v.bins), "rmse": v.rmse, **extra}


def entropy_case(data, name, params):
    c, z = data[name]
    try:
        v = skg.Variogram(c, z, n_lags=N_LAGS, fit_sigma="entropy", **params)
    except RuntimeError as e:
        return {"dataset": name, "params": params, "skgstat_fit_failed": type(e).__name__}
    return fitted(name, v, params, sigma=np.asarray(v.fit_sigma))


def ml_case(data, name, params):
    c, z = data[name]
    try:
        v = skg.Variogram(c, z, n_lags=N_LAGS, fit_method="ml", **params)
    except Exception as e:  # SLSQP failure raises OptimizeWarning
        return {"dataset": name, "params": params, "skgstat_fit_failed": type(e).__name__}
    keep = ~np.isnan(v.experimental)
    sigma = np.ones(v.bins.size) if v.fit_sigma is None else np.asarray(v.fit_sigma)
    return fitted(name, v, params, sigma=sigma[keep])


def harmonize_case(data, name, params):
    c, z = data[name]
    v = skg.Variogram(c, z, model="harmonize", **params)
    kept = v.bins[~np.isnan(v.experimental)]
    grid = np.linspace(0, kept.max(), 100)
    r, s, n = v.cof
    h = np.concatenate(([0.0], v.bins * 0.75, v.bins, [v.bins[-1] * 1.1]))
    return {"dataset": name, "params": params, "bins": v.bins, "experimental": v.experimental,
            "model_at_bins": v.transform(v.bins), "h": h, "model_at_h": v.transform(h),
            "skgstat_cof": [int(r), float(s), float(n)], "range": float(grid[int(r)]),
            "sill": float(s), "nugget": float(n), "rmse": v.rmse, "nse": v.NS}


SUMS = [["spherical", "gaussian"], ["exponential", "exponential"], ["spherical", "exponential"],
        ["gaussian", "cubic"], ["stable", "spherical"], ["spherical", "matern"],
        ["exponential", "spherical", "gaussian"]]


def sum_model_case(v, names, args):
    h = np.linspace(0, 2000, 41)
    return {"model": names, "args": list(args), "h": h,
            "values": v._build_sum_models("+".join(names))(h, *args)}


def sums_case(data, name, names, nugget):
    c, z = data[name]
    params = {"model": names, "use_nugget": nugget}
    try:
        v = skg.Variogram(c, z, n_lags=N_LAGS, model="+".join(names), use_nugget=nugget)
    except RuntimeError as e:
        return {"dataset": name, "params": params, "skgstat_fit_failed": type(e).__name__}
    return fitted(name, v, params)


def main():
    data = datasets()
    fitted_sets = ["meuse_lead", "meuse_zinc", "pancake"]
    entropy = [
        entropy_case(data, name, {"model": model, "use_nugget": nugget})
        for name in fitted_sets
        for model in ["spherical", "exponential", "gaussian", "matern"]
        for nugget in [False, True]
    ]
    ml = [
        ml_case(data, name, {"model": model, "use_nugget": nugget, **sigma})
        for name in fitted_sets
        for model in ["spherical", "exponential", "gaussian", "stable"]
        for nugget in [False, True]
        for sigma in [{}, {"fit_sigma": "linear"}]
    ]
    harmonize = [
        harmonize_case(data, name, {"n_lags": n_lags, "use_nugget": nugget})
        for name in data
        for n_lags in [8, 12]
        for nugget in [False, True]
    ]
    c, z = data["meuse_zinc"]
    probe = skg.Variogram(c, z, n_lags=N_LAGS, fit_method=None)
    sum_models = [
        sum_model_case(probe, ["spherical", "stable"], [300.0, 1.0, 800.0, 2.0, 1.5, 0.25]),
        sum_model_case(probe, ["spherical", "stable"], [300.0, 1.0, 800.0, 2.0, 1.5]),
        sum_model_case(probe, ["exponential", "gaussian"], [150.0, 2.5, 1200.0, 4.0, 0.5]),
        sum_model_case(probe, ["matern", "cubic"], [400.0, 3.0, 0.8, 1500.0, 1.0, 0.1]),
        sum_model_case(probe, ["spherical", "exponential", "gaussian"],
                       [100.0, 1.0, 600.0, 2.0, 1600.0, 3.0, 0.3]),
    ]
    sums = [sums_case(data, name, names, nugget)
            for name in fitted_sets for names in SUMS for nugget in [False, True]]
    dump_json("variogram/fit_extra.json", {
        "name": "variogram_fit_extra",
        "description": "scikit-gstat fitting options added in M10",
        "meta": meta(),
        "datasets": {k: {"coordinates": c.tolist(), "values": z.tolist()}
                     for k, (c, z) in data.items()},
        "entropy": entropy,
        "ml": ml,
        "harmonize": harmonize,
        "sum_models": sum_models,
        "sums": sums,
    })
    print(len(harmonize), "harmonize cases,", len(sum_models), "sum model evaluations")
    for section, cases in [("entropy", entropy), ("ml", ml), ("sums", sums)]:
        failed = sum("skgstat_fit_failed" in c for c in cases)
        print(len(cases), section, "cases,", failed, "failed in scikit-gstat")


if __name__ == "__main__":
    main()
