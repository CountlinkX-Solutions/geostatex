"""Generate scikit-gstat directional variogram fixtures (`fixtures/variogram/directional.json`).

Run from the repository root (under a memory cap):

    prlimit --as=3221225472 .venv/bin/python -I test/reference/generate_variogram_directional.py

The datasets of `generate_variogram_experimental.py` (Meuse lead and zinc, the pancake sample and
the synthetic set with duplicate points). Each case records skgstat.DirectionalVariogram's bins,
experimental semivariance, bin counts, n_lags and maxlag for one azimuth, tolerance, bandwidth and
directional model, with other lag and estimator options; some cases also fit a model (TRF), with
the fitted curve at the bins and the RMSE.
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import skgstat as skg  # noqa: E402

from common import dump_json, meta  # noqa: E402
from generate_variogram_experimental import datasets  # noqa: E402

warnings.simplefilter("ignore")


def record(name, coords, values, params, fit=None):
    kwargs = dict(params)
    if fit is None:
        # DirectionalVariogram always fits; a manual fit leaves the experimental part alone.
        kwargs.update({"fit_method": "manual", "fit_range": 1.0, "fit_sill": 1.0})
    else:
        kwargs.update(fit)
    v = skg.DirectionalVariogram(coords, values, **kwargs)
    out = {
        "dataset": name,
        "params": params,
        "bins": v.bins,
        "experimental": v.experimental,
        "bin_count": v.bin_count,
        "n_lags": int(v.n_lags),
        "maxlag": None if v.maxlag is None else float(v.maxlag),
        "bandwidth": float(v.bandwidth),
    }
    if fit is not None:
        out.update({"fit": fit, "cof": list(v.cof), "model_at_bins": v.transform(v.bins), "rmse": v.rmse})
    return out


def main():
    data = datasets()
    cases = []
    for name, (c, z) in data.items():
        for az in [0, 45, 90, -30, 135, 180, -180]:
            cases.append(record(name, c, z, {"azimuth": az, "n_lags": 10}))
        for tol in [10.0, 20.0, 90.0, 180.0, 360.0]:
            cases.append(record(name, c, z, {"azimuth": 60, "tolerance": tol, "n_lags": 8}))
        for bw in ["q10", "q50", "q90"]:
            cases.append(record(name, c, z, {"azimuth": -45, "bandwidth": bw, "n_lags": 8}))
        for model in ["triangle", "compass"]:
            cases.append(record(name, c, z, {"azimuth": 30, "tolerance": 30.0,
                                             "directional_model": model, "n_lags": 10}))
        for bf in ["uniform", "fd", "sturges"]:
            cases.append(record(name, c, z, {"azimuth": 90, "bin_func": bf, "n_lags": 6}))
        for ml in [0.5, "median", "mean"]:
            cases.append(record(name, c, z, {"azimuth": 0, "maxlag": ml, "n_lags": 8}))
        for est in ["cressie", "dowd", "genton"]:
            cases.append(record(name, c, z, {"azimuth": 120, "estimator": est, "n_lags": 8}))
        cases.append(record(name, c, z, {"azimuth": 45, "dist_func": "cityblock", "n_lags": 8}))
        for model in ["spherical", "exponential"]:
            cases.append(record(name, c, z, {"azimuth": 0, "tolerance": 60.0, "n_lags": 10},
                                fit={"model": model, "fit_method": "trf"}))
    # Explicit bandwidths, below each dataset's largest distance.
    for name, bw in [("meuse_lead", 300.0), ("meuse_zinc", 800.0), ("pancake", 40.0),
                     ("synthetic_dups", 15.0)]:
        c, z = data[name]
        cases.append(record(name, c, z, {"azimuth": 10, "bandwidth": bw, "n_lags": 8}))
        cases.append(record(name, c, z, {"azimuth": 10, "bandwidth": bw, "tolerance": 90.0,
                                         "directional_model": "compass", "n_lags": 8}))
    dump_json("variogram/directional.json", {
        "name": "variogram_directional",
        "description": "skgstat.DirectionalVariogram experimental semivariances and fits",
        "meta": meta(),
        "datasets": {k: {"coordinates": c.tolist(), "values": z.tolist()} for k, (c, z) in data.items()},
        "cases": cases,
    })
    print(len(cases), "cases")


if __name__ == "__main__":
    main()
