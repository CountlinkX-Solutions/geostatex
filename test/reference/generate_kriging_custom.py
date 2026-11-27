"""Generate PyKrige custom-variogram fixtures (`fixtures/kriging/custom.json`).

Run from the repository root (under a memory cap):

    prlimit --as=3221225472 .venv/bin/python -I test/reference/generate_kriging_custom.py

PyKrige's `variogram_model="custom"` takes `variogram_parameters` (a list) and
`variogram_function(params, distances)`, which it applies to whole distance matrices. The two
functions below are named in each case; the Elixir test defines the same ones.

Datasets: Meuse (log zinc, metres), a seeded geographic set in degrees and a seeded 3D set.
Cases cover the styles, the moving window, anisotropy, exact values, the pseudo-inverse,
geographic coordinates, universal kriging with a regional linear drift, 3D ordinary kriging and
PyKrige's statistics.
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import skgstat as skg  # noqa: E402
from pykrige.ok import OrdinaryKriging  # noqa: E402
from pykrige.ok3d import OrdinaryKriging3D  # noqa: E402
from pykrige.uk import UniversalKriging  # noqa: E402

from common import dump_json, meta  # noqa: E402

warnings.simplefilter("ignore")

FUNCTIONS = {
    # A stable model: sill * (1 - exp(-(d / range) ^ shape)) + nugget.
    "stable": lambda p, d: p[0] * (1.0 - np.exp(-((d / p[1]) ** p[2]))) + p[3],
    # A damped hole effect: sill * (1 - cos(d / a) * exp(-d / b)) + nugget.
    "hole": lambda p, d: p[0] * (1.0 - np.cos(d / p[1]) * np.exp(-d / p[2])) + p[3],
}


def datasets():
    out = {}
    c, z = skg.data.meuse(variable="zinc").get("sample")
    out["meuse_logzinc"] = (np.asarray(c, float), np.log(np.asarray(z, float).ravel()))
    rng = np.random.default_rng(3)
    lon = rng.uniform(-10, 10, 40)
    lat = rng.uniform(35, 50, 40)
    out["geo"] = (np.column_stack([lon, lat]), 2 + 0.1 * lon + 0.05 * lat + rng.normal(0, 0.2, 40))
    p = rng.uniform(0, 10, (50, 3))
    out["cube"] = (p, np.sin(p[:, 0]) + 0.3 * p[:, 1] - 0.2 * p[:, 2] + rng.normal(0, 0.1, 50))
    return out


def targets(c, style):
    rng = np.random.default_rng(11)
    if style == "points":
        idx = rng.choice(len(c), 4, replace=False)
        cols = [np.concatenate([rng.uniform(c[:, k].min(), c[:, k].max(), 8), c[idx, k]])
                for k in range(c.shape[1])]
        return {"coords": [col.tolist() for col in cols]}
    xs = np.linspace(c[:, 0].min(), c[:, 0].max(), 7)
    ys = np.linspace(c[:, 1].min(), c[:, 1].max(), 5)
    out = {"xs": xs.tolist(), "ys": ys.tolist()}
    if style == "masked":
        out["mask"] = (np.add.outer(np.arange(5), np.arange(7)) % 4 == 0).tolist()
    return out


def run(data, name, kind, function, params, style="points", init=None, execute=None,
        statistics=False):
    c, z = data[name]
    init = dict(init or {})
    custom = dict(variogram_model="custom", variogram_parameters=params,
                  variogram_function=FUNCTIONS[function])
    if kind == "ordinary":
        k = OrdinaryKriging(c[:, 0], c[:, 1], z, enable_statistics=statistics, **custom, **init)
    elif kind == "universal":
        k = UniversalKriging(c[:, 0], c[:, 1], z, **custom, **init)
    else:
        k = OrdinaryKriging3D(c[:, 0], c[:, 1], c[:, 2], z, **custom, **init)
    where = targets(c, style)
    execute = dict(execute or {})
    if style == "points":
        zv, ss = k.execute("points", *[np.asarray(col) for col in where["coords"]], **execute)
    else:
        mask = np.asarray(where["mask"]) if style == "masked" else None
        args = [np.asarray(where["xs"]), np.asarray(where["ys"])]
        if mask is not None:
            execute["mask"] = mask
        zv, ss = k.execute(style, *args, **execute)
    case = {"dataset": name, "kind": kind, "function": function, "params": params,
            "init": init, "style": style, "targets": where,
            "execute": {k_: v for k_, v in execute.items() if k_ != "mask"},
            "z": np.ma.filled(np.ma.asarray(zv, dtype=float), np.nan),
            "ss": np.ma.filled(np.ma.asarray(ss, dtype=float), np.nan),
            "lags": k.lags, "semivariance": k.semivariance}
    if statistics:
        q1, q2, cr = k.get_statistics()
        case["statistics"] = {"delta": k.delta, "sigma": k.sigma, "epsilon": k.epsilon,
                              "q1": q1, "q2": q2, "cr": cr}
    return case


def main():
    data = datasets()
    st = ("stable", [0.6, 900.0, 1.4, 0.05])
    hole = ("hole", [0.55, 300.0, 1500.0, 0.04])
    cases = []
    for function, params in [st, hole]:
        for style in ["points", "grid", "masked"]:
            cases.append(run(data, "meuse_logzinc", "ordinary", function, params, style))
    cases += [
        run(data, "meuse_logzinc", "ordinary", *st, "grid",
            execute={"n_closest_points": 10, "backend": "loop"}),
        run(data, "meuse_logzinc", "ordinary", *hole, "points",
            execute={"n_closest_points": 8, "backend": "loop"}),
        run(data, "meuse_logzinc", "ordinary", *st, "grid",
            init={"anisotropy_scaling": 2.0, "anisotropy_angle": 30.0}),
        run(data, "meuse_logzinc", "ordinary", *st, "points", init={"exact_values": False}),
        run(data, "meuse_logzinc", "ordinary", *hole, "points", init={"pseudo_inv": True}),
        run(data, "meuse_logzinc", "ordinary", *st, "points", statistics=True),
        run(data, "meuse_logzinc", "ordinary", *hole, "points", statistics=True),
        run(data, "geo", "ordinary", "stable", [0.15, 8.0, 1.2, 0.01], "grid",
            init={"coordinates_type": "geographic"}),
        run(data, "meuse_logzinc", "universal", *st, "points",
            init={"drift_terms": ["regional_linear"]}),
        run(data, "meuse_logzinc", "universal", *hole, "grid",
            init={"drift_terms": ["regional_linear"]}),
        run(data, "cube", "ordinary_3d", "stable", [0.8, 6.0, 1.3, 0.02], "points"),
    ]
    dump_json("kriging/custom.json", {
        "name": "kriging_custom",
        "description": "PyKrige variogram_model='custom' with two named functions",
        "meta": meta(),
        "datasets": {k: {"coordinates": c.tolist(), "values": z.tolist()}
                     for k, (c, z) in data.items()},
        "cases": cases,
    })
    print(len(cases), "cases")


if __name__ == "__main__":
    main()
