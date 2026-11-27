"""Generate scikit-gstat experimental variogram fixtures (`fixtures/variogram/experimental.json`).

Run from the repository root (under a memory cap):

    prlimit --as=3221225472 .venv/bin/python -I test/reference/generate_variogram_experimental.py

Datasets: Meuse topsoil heavy metals (lead, zinc; Pebesma and Bivand 2005, bundled with
scikit-gstat), the pancake sample (seed 42, 150 points) and a seeded synthetic set with duplicate
points (distance 0). Each case records skgstat's bins (upper edges), experimental semivariance,
bin counts, n_lags and maxlag for one combination of estimator, lag rule, maxlag and metric.
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import skgstat as skg  # noqa: E402

from common import dump_json, meta  # noqa: E402

warnings.simplefilter("ignore")

# scipy pdist metrics; the extra ones (M9) take scipy's defaults (seuclidean V and mahalanobis
# VI from the coordinates, minkowski p = 2), as scikit-gstat passes no keyword arguments.
# "correlation" is left out: on 2D coordinates it is 0 or 2, and rounding decides the last lag.
METRICS = ["cityblock", "chebyshev", "sqeuclidean", "seuclidean", "canberra", "braycurtis",
           "cosine", "mahalanobis", "minkowski"]


def datasets():
    out = {}
    for var in ("lead", "zinc"):
        c, z = skg.data.meuse(variable=var).get("sample")
        out[f"meuse_{var}"] = (np.asarray(c, float), np.asarray(z, float).ravel())
    c, z = skg.data.pancake(N=150, seed=42).get("sample")
    out["pancake"] = (np.asarray(c, float), np.asarray(z, float).ravel())
    rng = np.random.default_rng(7)
    c = rng.uniform(0, 100, size=(60, 2))
    c[10] = c[3]
    c[25] = c[3]
    z = 5 + 0.05 * c[:, 0] + rng.normal(0, 1, 60)
    out["synthetic_dups"] = (c, z)
    return out


def record(name, coords, values, params):
    v = skg.Variogram(coords, values, fit_method=None, **params)
    maxlag = v.maxlag
    return {
        "dataset": name,
        "params": {k: (list(val) if isinstance(val, (list, tuple, np.ndarray)) else val)
                   for k, val in params.items()},
        "bins": v.bins,
        "experimental": v.experimental,
        "bin_count": v.bin_count,
        "n_lags": int(v.n_lags),
        "maxlag": None if maxlag is None else float(maxlag),
    }


def main():
    data = datasets()
    cases = []
    estimators = ["matheron", "cressie", "dowd", "genton", "minmax", "percentile", "entropy"]
    for name, (c, z) in data.items():
        for est in estimators:
            cases.append(record(name, c, z, {"estimator": est, "n_lags": 12}))
        for bf in ["even", "uniform", "sturges", "scott", "sqrt", "fd", "doane", "rice"]:
            cases.append(record(name, c, z, {"bin_func": bf, "n_lags": 8}))
        for ml in [None, 0.5, "mean", "median", 300.0]:
            cases.append(record(name, c, z, {"maxlag": ml, "n_lags": 10}))
        for ml in [0.5, "median"]:
            cases.append(record(name, c, z, {"maxlag": ml, "bin_func": "uniform", "n_lags": 6}))
            cases.append(record(name, c, z, {"maxlag": ml, "bin_func": "fd"}))
        for metric in METRICS:
            cases.append(record(name, c, z, {"dist_func": metric, "n_lags": 10}))
    c, z = data["pancake"]
    cases.append(record("pancake", c, z, {"estimator": "percentile", "percentile": 80, "n_lags": 8}))
    cases.append(record("pancake", c, z, {"estimator": "entropy", "entropy_bins": 20, "n_lags": 8}))
    cases.append(record("pancake", c, z, {"bin_func": [20.0, 50.0, 90.0, 140.0, 200.0]}))
    dump_json("variogram/experimental.json", {
        "name": "variogram_experimental",
        "description": "skgstat.Variogram experimental semivariances",
        "meta": meta(),
        "datasets": {k: {"coordinates": c.tolist(), "values": z.tolist()} for k, (c, z) in data.items()},
        "cases": cases,
    })
    print(len(cases), "cases")


if __name__ == "__main__":
    main()
