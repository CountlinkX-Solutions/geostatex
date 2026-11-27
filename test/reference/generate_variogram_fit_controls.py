"""Generate scikit-gstat fit control fixtures (`fixtures/variogram/fit_controls.json`).

Run from the repository root (under a memory cap):

    prlimit --as=3221225472 .venv/bin/python -I test/reference/generate_variogram_fit_controls.py

`skgstat.Variogram(..., fit_bounds=, fit_p0=)` (scikit-gstat 1.0.12+): the bounds and the
initial guess passed to `scipy.optimize.curve_fit` for `trf` (both) and `lm` (`p0` only), and to
the maximum-likelihood fit (`p0`, which also sets its upper bounds). Cases where a bound binds,
lower bounds above zero, a start away from the default, a model with a shape, with and without
nugget. The coefficients, parameters and goodness of fit are recorded.
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import skgstat as skg  # noqa: E402

from common import dump_json, meta  # noqa: E402
from generate_variogram_fit import datasets  # noqa: E402

warnings.simplefilter("ignore")


def record(name, c, z, params):
    v = skg.Variogram(c, z, n_lags=12, **params)
    out = {"dataset": name, "params": params, "bins": v.bins, "experimental": v.experimental,
           "cof": list(v.cof), "parameters": v.parameters, "rmse": v.rmse}
    print(name, params.get("model"), params.get("fit_method", "trf"), [round(x, 4) for x in v.cof])
    return out


def main():
    data = datasets()
    c, z = data["pancake"]
    cl, zl = data["meuse_lead"]
    cases = [
        # The range bound binds (below the free optimum).
        record("pancake", c, z, {"model": "spherical", "fit_bounds": ([0, 0], [200.0, 2000.0])}),
        # Lower bounds above zero, with a nugget.
        record("pancake", c, z, {"model": "exponential", "use_nugget": True,
                                 "fit_bounds": ([100.0, 500.0, 50.0], [600.0, 2000.0, 400.0])}),
        # A start away from the default (the upper bounds).
        record("pancake", c, z, {"model": "spherical", "use_nugget": True,
                                 "fit_p0": [150.0, 900.0, 100.0]}),
        # Both, with a shape.
        record("meuse_lead", cl, zl, {"model": "stable", "use_nugget": True,
                                      "fit_bounds": ([0, 0, 0.5, 0], [1500.0, 20000.0, 1.8, 5000.0]),
                                      "fit_p0": [800.0, 10000.0, 1.0, 1000.0]}),
        # Levenberg-Marquardt: p0 only (no bounds).
        record("pancake", c, z, {"model": "gaussian", "fit_method": "lm",
                                 "fit_p0": [300.0, 1200.0]}),
        # Maximum likelihood: p0 also sets the upper bounds (0, p0). scikit-gstat needs it as an
        # array there (it multiplies it by 0.5).
        record("pancake", c, z, {"model": "spherical", "fit_method": "ml",
                                 "fit_p0": np.array([250.0, 1500.0])}),
    ]
    dump_json("variogram/fit_controls.json", {
        "name": "variogram_fit_controls",
        "description": "skgstat Variogram fits with fit_bounds and fit_p0",
        "meta": meta(),
        "datasets": {k: {"coordinates": d[0].tolist(), "values": d[1].tolist()}
                     for k, d in data.items() if k in ("pancake", "meuse_lead")},
        "cases": cases,
    })
    print(len(cases), "cases")


if __name__ == "__main__":
    main()
