"""Generate cross-variogram fixtures (`fixtures/variogram/cross.json`).

Run from the repository root (under a memory cap):

    prlimit --as=3221225472 .venv/bin/python -I test/reference/generate_variogram_cross.py

scikit-gstat 1.0.24 computes a cross-variogram from `values` of shape `(n, 2)`: the differences
are `|a_i - a_j| * |b_i - b_j|` and go through the estimator (`skgstat` cases).
`skgstat.util.cross_variograms` builds the grid of every pair of variables (`grids`).

The classical cross-semivariogram `sum((a_i - a_j) (b_i - b_j)) / 2N` is not in scikit-gstat;
its cases (`classical`) are computed here with numpy over the lag classes scikit-gstat assigns to
the pairs (`Variogram.lag_groups`).

Datasets: Meuse topsoil lead, zinc and copper (Pebesma and Bivand 2005, bundled with
scikit-gstat) and a seeded synthetic set with a positively (`b`) and a negatively (`c`)
correlated co-variable.
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import skgstat as skg  # noqa: E402
from skgstat.util.cross_variogram import cross_variograms  # noqa: E402

from common import dump_json, meta  # noqa: E402

warnings.simplefilter("ignore")


def datasets():
    meuse = {}
    for var in ("lead", "zinc", "copper"):
        c, z = skg.data.meuse(variable=var).get("sample")
        meuse[var] = np.asarray(z, float).ravel()
    rng = np.random.default_rng(11)
    sc = rng.uniform(0, 100, size=(80, 2))
    a = 0.04 * sc[:, 0] + rng.normal(0, 1, 80)
    synthetic = {
        "a": a,
        "b": 0.8 * a + rng.normal(0, 0.5, 80),
        "c": -0.8 * a + rng.normal(0, 0.5, 80),
    }
    return {
        "meuse": (np.asarray(c, float), meuse),
        "synthetic": (sc, synthetic),
    }


def summary(v):
    return {
        "bins": v.bins,
        "experimental": v.experimental,
        "bin_count": v.bin_count,
        "n_lags": int(v.n_lags),
        "maxlag": None if v.maxlag is None else float(v.maxlag),
    }


def plain(params):
    return {k: (list(val) if isinstance(val, (list, tuple, np.ndarray)) else val)
            for k, val in params.items()}


def skgstat_case(data, name, primary, co, params, directional=False):
    coords, values = data[name]
    stacked = np.column_stack((values[primary], values[co]))
    if directional:
        # DirectionalVariogram refuses fit_method=None; only the experimental values are kept.
        v = skg.DirectionalVariogram(coords, stacked, **params)
    else:
        v = skg.Variogram(coords, stacked, fit_method=None, **params)
    assert v.is_cross_variogram
    return {"dataset": name, "primary": primary, "co": co, "cross": "skgstat",
            "params": plain(params), **summary(v)}


def classical_case(data, name, primary, co, params):
    coords, values = data[name]
    a, b = values[primary], values[co]
    v = skg.Variogram(coords, a, fit_method=None, **params)
    groups = v.lag_groups()
    iu = np.triu_indices(len(a), 1)
    da = (a[:, None] - a[None, :])[iu]
    db = (b[:, None] - b[None, :])[iu]
    products = da * db
    experimental = []
    for k in range(len(v.bins)):
        selected = products[groups == k]
        experimental.append(selected.sum() / (2 * len(selected)) if len(selected) else np.nan)
    out = summary(v)
    out["experimental"] = np.asarray(experimental)
    return {"dataset": name, "primary": primary, "co": co, "cross": "classical",
            "params": plain(params), **out}


def grid_case(data, name, names, params):
    coords, values = data[name]
    stacked = np.column_stack([values[n] for n in names])
    grid = cross_variograms(coords, stacked, fit_method=None, **params)
    return {
        "dataset": name,
        "variables": names,
        "params": plain(params),
        "cells": [[{"cross": v.is_cross_variogram, **summary(v)} for v in row] for row in grid],
    }


def main():
    data = datasets()
    cases = []
    for est in ["matheron", "cressie", "dowd", "genton", "minmax", "percentile", "entropy"]:
        cases.append(skgstat_case(data, "meuse", "lead", "zinc", {"estimator": est, "n_lags": 8}))
    for params in [{"bin_func": "uniform", "n_lags": 6}, {"bin_func": "fd"},
                   {"maxlag": 0.5, "n_lags": 10}, {"maxlag": "median", "n_lags": 6},
                   {"dist_func": "cityblock", "n_lags": 8}]:
        cases.append(skgstat_case(data, "meuse", "zinc", "copper", params))
    cases.append(skgstat_case(data, "meuse", "lead", "zinc",
                              {"azimuth": 30, "tolerance": 40.0, "n_lags": 6}, directional=True))
    for co in ["b", "c"]:
        cases.append(skgstat_case(data, "synthetic", "a", co, {"n_lags": 10}))
        cases.append(skgstat_case(data, "synthetic", "a", co,
                                  {"estimator": "cressie", "n_lags": 10}))

    for primary, co in [("lead", "zinc"), ("zinc", "copper")]:
        cases.append(classical_case(data, "meuse", primary, co, {"n_lags": 8}))
    for co in ["b", "c"]:
        cases.append(classical_case(data, "synthetic", "a", co, {"n_lags": 10}))
        cases.append(classical_case(data, "synthetic", "a", co,
                                    {"maxlag": 0.5, "bin_func": "uniform", "n_lags": 6}))

    grids = [
        grid_case(data, "meuse", ["lead", "zinc", "copper"], {"n_lags": 5}),
        grid_case(data, "synthetic", ["a", "b", "c"], {"estimator": "cressie", "n_lags": 6}),
    ]

    dump_json("variogram/cross.json", {
        "name": "variogram_cross",
        "description": "skgstat cross-variograms, the classical definition and cross_variograms",
        "meta": meta(),
        "datasets": {
            name: {"coordinates": c.tolist(), "values": {k: v.tolist() for k, v in vals.items()}}
            for name, (c, vals) in data.items()
        },
        "cases": cases,
        "grids": grids,
    })
    print(len(cases), "cases,", len(grids), "grids")


if __name__ == "__main__":
    main()
