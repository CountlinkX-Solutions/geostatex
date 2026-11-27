"""Generate scikit-gstat variogram model and fit fixtures (`fixtures/variogram/fit.json`).

Run from the repository root (under a memory cap):

    prlimit --as=3221225472 .venv/bin/python -I test/reference/generate_variogram_fit.py

Two parts:

* ``models``: every model function evaluated by scikit-gstat on a dense lag grid with fixed
  parameters (tests the formulas, including the Matern Bessel function, exactly);
* ``cases``: fits of every model on Meuse (lead, zinc) and the pancake sample with ``trf``
  (with and without nugget, and with each ``fit_sigma``), ``lm`` and ``manual``, recording the
  coefficients, the parameters and the goodness of fit.
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import skgstat as skg  # noqa: E402
from skgstat import models  # noqa: E402

from common import dump_json, meta  # noqa: E402

warnings.simplefilter("ignore")

MODELS = ["spherical", "exponential", "gaussian", "cubic", "stable", "matern"]


def datasets():
    out = {}
    for var in ("lead", "zinc"):
        c, z = skg.data.meuse(variable=var).get("sample")
        out[f"meuse_{var}"] = (np.asarray(c, float), np.asarray(z, float).ravel())
    c, z = skg.data.pancake(N=150, seed=42).get("sample")
    out["pancake"] = (np.asarray(c, float), np.asarray(z, float).ravel())
    return out


def model_grid():
    h = np.linspace(0, 120, 61).tolist() + [0.001, 37.5, 49.999, 50.0, 50.001, 400.0]
    out = []
    for name in MODELS:
        f = getattr(models, name)
        params = {"r": 50.0, "c0": 3.0, "b": 0.5}
        shapes = [None]
        if name == "stable":
            shapes = [0.5, 1.0, 1.7, 2.0]
        if name == "matern":
            shapes = [0.3, 0.5, 1.5, 2.5, 7.0, 20.0]
        for s in shapes:
            if s is None:
                vals = [float(f(x, 50.0, 3.0, 0.5)) for x in h]
            else:
                vals = [float(f(x, 50.0, 3.0, s, 0.5)) for x in h]
            out.append({"model": name, "params": dict(params, s=s), "h": h, "values": vals})
    return out


def diagnostics(v):
    return {"rmse": v.rmse, "mse": v.mse, "mae": v.mae, "nrmse": v.nrmse,
            "nrmse_r": v.nrmse_r, "r": v.r, "nse": v.NS}


def record(name, c, z, params):
    try:
        v = skg.Variogram(c, z, n_lags=12, **params)
    except (ZeroDivisionError, RuntimeError) as e:
        # scikit-gstat's own fit failed: it stepped to a zero range (its model divides by zero)
        # or scipy's optimizer did not converge.
        print("fit failed in scikit-gstat:", name, params, type(e).__name__)
        return {"dataset": name, "params": params, "skgstat_fit_failed": type(e).__name__}
    out = {"dataset": name, "params": params, "bins": v.bins, "experimental": v.experimental,
           "cof": list(v.cof)}
    try:
        out["parameters"] = v.parameters
        out["model_at_bins"] = v.transform(v.bins)
        out.update(diagnostics(v))
    except ZeroDivisionError:
        # The fit collapsed to a zero range: scikit-gstat cannot evaluate its own model.
        out["degenerate"] = True
        print("degenerate:", name, params, list(v.cof))
    return out


def main():
    data = datasets()
    cases = []
    for name, (c, z) in data.items():
        for m in MODELS:
            cases.append(record(name, c, z, {"model": m}))
            cases.append(record(name, c, z, {"model": m, "use_nugget": True}))
        for sigma in ["linear", "exp", "sqrt", "sq"]:
            cases.append(record(name, c, z, {"model": "spherical", "fit_sigma": sigma, "use_nugget": True}))
        for m in ["exponential", "gaussian"]:
            cases.append(record(name, c, z, {"model": m, "fit_method": "lm"}))
    c, z = data["pancake"]
    cases.append(record("pancake", c, z, {"model": "spherical", "fit_method": "manual",
                                           "fit_range": 300.0, "fit_sill": 1200.0, "fit_nugget": 50.0,
                                           "use_nugget": True}))
    cases.append(record("pancake", c, z, {"model": "stable", "fit_method": "manual",
                                           "fit_range": 250.0, "fit_sill": 1100.0, "fit_shape": 1.5}))
    dump_json("variogram/fit.json", {
        "name": "variogram_fit",
        "description": "skgstat models and Variogram fits",
        "meta": meta(),
        "datasets": {k: {"coordinates": c.tolist(), "values": z.tolist()} for k, (c, z) in data.items()},
        "models": model_grid(),
        "cases": cases,
    })
    print(len(cases), "cases")


if __name__ == "__main__":
    main()
