"""Generate indicator kriging fixtures (`fixtures/kriging/indicator.json`).

Run from the repository root (under a memory cap):

    prlimit --as=3221225472 .venv/bin/python -I test/reference/generate_kriging_indicator.py

PyKrige has no indicator kriging class: the reference is `pykrige.OrdinaryKriging` on the
indicators `z <= t` of each threshold. Meuse log-zinc (155 points), thresholds at the quartiles;
cases with given spherical parameters per threshold, one set for every threshold, an automatic
fit per threshold, and an exponential model on a grid. The raw (uncorrected) kriged indicators
are stored per threshold.
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from pykrige.ok import OrdinaryKriging  # noqa: E402

from common import dump_json, meta  # noqa: E402
from generate_kriging_ordinary import datasets  # noqa: E402

warnings.simplefilter("ignore")


def main():
    c, z = datasets()["meuse_logzinc"]
    x, y = c[:, 0], c[:, 1]
    thresholds = list(np.quantile(z, [0.25, 0.5, 0.75]))
    rng = np.random.default_rng(14)
    px = np.concatenate([rng.uniform(x.min(), x.max(), 10), x[:2]])
    py = np.concatenate([rng.uniform(y.min(), y.max(), 10), y[:2]])
    gx, gy = np.linspace(x.min(), x.max(), 6), np.linspace(y.min(), y.max(), 5)
    cases = []

    def run(name, model, params, style="points", nlags=6):
        raw = []
        fitted = []
        for k, t in enumerate(thresholds):
            ind = (z <= t).astype(float)
            init = {"variogram_model": model, "nlags": nlags}
            if params is not None:
                init["variogram_parameters"] = params[k] if isinstance(params[0], list) else params
            ok = OrdinaryKriging(x, y, ind, **init)
            fitted.append(list(ok.variogram_model_parameters))
            if style == "points":
                p, _ = ok.execute("points", px, py)
            else:
                p, _ = ok.execute("grid", gx, gy)
            raw.append(np.asarray(p).ravel())
        where = {"x": px, "y": py} if style == "points" else {"x": gx, "y": gy}
        cases.append({"name": name, "variogram_model": model, "parameters": params,
                      "nlags": nlags, "fitted": fitted, "style": style, "where": where,
                      "raw": raw})

    per_threshold = [[0.12, 900.0, 0.05], [0.15, 1000.0, 0.06], [0.11, 800.0, 0.05]]
    run("per_threshold", "spherical", per_threshold)
    run("per_threshold_grid", "spherical", per_threshold, style="grid")
    run("one_for_all", "spherical", [0.14, 950.0, 0.05])
    run("automatic", "spherical", None)
    run("exponential_grid", "exponential", [0.15, 1200.0, 0.04], style="grid")
    dump_json("kriging/indicator.json", {
        "name": "kriging_indicator",
        "description": "pykrige OrdinaryKriging of the indicators z <= t at the quartiles",
        "meta": meta(),
        "data": {"x": x, "y": y, "values": z},
        "thresholds": thresholds,
        "cases": cases,
    })
    print(len(cases), "cases")


if __name__ == "__main__":
    main()
