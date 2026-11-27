"""Generate scikit-gstat space-time variogram fixtures (`fixtures/variogram/spacetime.json`).

Run from the repository root (under a memory cap):

    prlimit --as=3221225472 .venv/bin/python -I test/reference/generate_variogram_spacetime.py

Two seeded synthetic space-time fields (points by time steps: a spatial trend, a seasonal term
and autoregressive noise). Each case records skgstat.SpaceTimeVariogram's space and time lag
edges, the experimental grid (space lags by time lags), both marginal variograms (bins,
experimental, fitted coefficients) and the space-time model: the fixed sills and, for the
product-sum model, its coefficients.

scikit-gstat fits the product-sum model against a transposed lag grid (`meshgrid` flattens time
major, the experimental grid is space major). The fixture records that fit as `skgstat_cof` and
the same TRF fit (`scipy.optimize.curve_fit`, bounds [0, inf), p0 = 1) on the aligned grid as
`cof`, which Geostatex reproduces.
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import skgstat as skg  # noqa: E402
from scipy.optimize import curve_fit  # noqa: E402

from common import dump_json, meta  # noqa: E402

warnings.simplefilter("ignore")


def field(seed, n_points, n_times):
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 100, size=(n_points, 2))
    v = np.empty((n_points, n_times))
    noise = rng.normal(0, 1, n_points)
    for t in range(n_times):
        noise = 0.6 * noise + rng.normal(0, 0.8, n_points)
        v[:, t] = 0.04 * coords[:, 0] + 2 * np.sin(t / 2) + noise
    return coords, v


def marginal(var):
    return {"bins": var.bins, "experimental": var.experimental, "cof": list(var.cof),
            "n_lags": int(var.n_lags)}


def record(name, coords, values, params):
    st = skg.SpaceTimeVariogram(coords, values, **params)
    st.fit(force=True)
    xm, tm = st.XMarginal, st.TMarginal
    out = {
        "dataset": name, "params": params,
        "xbins": [float(b) for b in st.xbins], "tbins": [float(b) for b in st.tbins],
        "x_lags": int(st.x_lags), "t_lags": int(st.t_lags),
        "experimental": [float(v) for v in st.experimental],
        "x_marginal": marginal(xm), "t_marginal": marginal(tm),
        "model_params": {k: float(v) for k, v in st._model_params.items()},
        "skgstat_cof": [float(c) for c in st.cof],
    }
    model = params.get("model", "product-sum")
    if model in ("product-sum", "product_sum"):
        Vx, Vt = xm.fitted_model, tm.fitted_model
        z = st.experimental
        keep = ~np.isnan(z)
        h, t = np.meshgrid(st.xbins, st.tbins, indexing="ij")
        lags = np.vstack((h.flatten()[keep], t.flatten()[keep])).T
        Cx, Ct = out["model_params"]["Cx"], out["model_params"]["Ct"]

        def f(lags, k1, k2, k3):
            return np.array([(k2 + k1 * Ct) * Vx(a) + (k3 + k1 * Cx) * Vt(b) - k1 * Vx(a) * Vt(b)
                             for a, b in lags], dtype=float)

        cof, _ = curve_fit(f, lags, z[keep], bounds=[0, np.inf], p0=[1.0] * 3)
        out["cof"] = [float(c) for c in cof]
    return out


def main():
    data = {"small": field(3, 25, 8), "medium": field(5, 30, 10)}
    cases = []
    for name, (c, v) in data.items():
        for model in ["sum", "product", "product-sum"]:
            cases.append(record(name, c, v, {"model": model}))
        for est in ["cressie", "dowd", "minmax"]:
            cases.append(record(name, c, v, {"estimator": est, "x_lags": 8}))
        for xbins in ["uniform", "sturges", "fd"]:
            cases.append(record(name, c, v, {"xbins": xbins, "x_lags": 6}))
        cases.append(record(name, c, v, {"tbins": "uniform", "t_lags": 4}))
        cases.append(record(name, c, v, {"t_lags": 3, "x_lags": 5}))
        for ml in [0.6, "median", 40.0]:
            cases.append(record(name, c, v, {"maxlag": ml, "x_lags": 7}))
        cases.append(record(name, c, v, {"use_nugget": True, "x_lags": 6}))
        cases.append(record(name, c, v, {"xdist_func": "cityblock", "x_lags": 6}))
    dump_json("variogram/spacetime.json", {
        "name": "variogram_spacetime",
        "description": "skgstat.SpaceTimeVariogram experimental grids, marginals and models",
        "meta": meta(),
        "datasets": {k: {"coordinates": c.tolist(), "values": v.tolist()} for k, (c, v) in data.items()},
        "cases": cases,
    })
    print(len(cases), "cases")


if __name__ == "__main__":
    main()
