"""Generate PyKrige ordinary kriging fixtures (`fixtures/kriging/ordinary.json`).

Run from the repository root (under a memory cap):

    prlimit --as=3221225472 .venv/bin/python -I test/reference/generate_kriging_ordinary.py

Datasets: Meuse (log zinc, metres), the pancake sample, a seeded geographic set in degrees and a
small set with a duplicated point. ``explicit`` cases fix the variogram parameters (prediction is
then deterministic and compared exactly); ``auto`` cases let PyKrige fit the variogram.
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import skgstat as skg  # noqa: E402
from pykrige.ok import OrdinaryKriging  # noqa: E402

from common import dump_json, meta  # noqa: E402

warnings.simplefilter("ignore")


def datasets():
    out = {}
    c, z = skg.data.meuse(variable="zinc").get("sample")
    out["meuse_logzinc"] = (np.asarray(c, float), np.log(np.asarray(z, float).ravel()))
    c, z = skg.data.pancake(N=150, seed=42).get("sample")
    out["pancake"] = (np.asarray(c, float), np.asarray(z, float).ravel())
    rng = np.random.default_rng(3)
    lon = rng.uniform(-10, 10, 40)
    lat = rng.uniform(35, 50, 40)
    out["geo"] = (np.column_stack([lon, lat]), 2 + 0.1 * lon + 0.05 * lat + rng.normal(0, 0.2, 40))
    c = rng.uniform(0, 10, (12, 2))
    c[5] = c[2]
    out["duplicated"] = (c, rng.normal(10, 2, 12))
    return out


def grid_for(c, nx=9, ny=7):
    xs = np.linspace(c[:, 0].min(), c[:, 0].max(), nx)
    ys = np.linspace(c[:, 1].min(), c[:, 1].max(), ny)
    return xs, ys


def run(name, c, z, init, style="points", execute=None):
    ok = OrdinaryKriging(c[:, 0], c[:, 1], z, **init)
    execute = dict(execute or {})
    if style == "points":
        rng = np.random.default_rng(11)
        idx = rng.choice(len(z), 5, replace=False)
        px = np.concatenate([rng.uniform(c[:, 0].min(), c[:, 0].max(), 10), c[idx, 0]])
        py = np.concatenate([rng.uniform(c[:, 1].min(), c[:, 1].max(), 10), c[idx, 1]])
        zv, ss = ok.execute("points", px, py, **execute)
        where = {"x": px, "y": py}
    else:
        xs, ys = grid_for(c)
        if style == "masked":
            mask = np.zeros((len(ys), len(xs)), dtype=bool)
            mask[::2, 1::3] = True
            execute["mask"] = mask
            where = {"x": xs, "y": ys, "mask": mask.astype(int)}
        else:
            where = {"x": xs, "y": ys}
        zv, ss = ok.execute(style, xs, ys, **execute)
        zv, ss = np.ma.filled(zv, np.nan), np.ma.filled(ss, np.nan)
    init_json = {k: (list(v) if isinstance(v, (list, tuple, np.ndarray)) else v) for k, v in init.items()}
    return {
        "dataset": name, "init": init_json, "style": style,
        "execute": {k: v for k, v in execute.items() if k != "mask"},
        "where": where,
        "variogram_model_parameters": list(ok.variogram_model_parameters),
        "lags": ok.lags, "semivariance": ok.semivariance,
        "z": np.asarray(zv).ravel(), "sigmasq": np.asarray(ss).ravel(),
    }


def main():
    data = datasets()
    cases = []
    c, z = data["meuse_logzinc"]
    explicit = {
        "linear": [0.0006, 0.05], "power": [0.002, 1.3, 0.04],
        "gaussian": [0.7, 1100.0, 0.05], "spherical": [0.7, 1200.0, 0.05],
        "exponential": [0.7, 1500.0, 0.05], "hole-effect": [0.7, 900.0, 0.05],
    }
    for model, params in explicit.items():
        for style in ["points", "grid"]:
            cases.append(run("meuse_logzinc", c, z, {"variogram_model": model, "variogram_parameters": params}, style))
    sph = {"variogram_model": "spherical", "variogram_parameters": {"sill": 0.75, "range": 1200.0, "nugget": 0.05}}
    cases.append(run("meuse_logzinc", c, z, dict(sph, exact_values=False), "points"))
    cases.append(run("meuse_logzinc", c, z, dict(sph, anisotropy_scaling=2.0, anisotropy_angle=30.0), "grid"))
    cases.append(run("meuse_logzinc", c, z, sph, "masked"))
    cases.append(run("meuse_logzinc", c, z, sph, "grid", {"n_closest_points": 12, "backend": "loop"}))
    cases.append(run("meuse_logzinc", c, z, {"variogram_model": "spherical", "variogram_parameters": {"psill": 0.7, "range": 1200.0, "nugget": 0.05}}, "points"))
    gc, gz = data["geo"]
    geo = {"variogram_model": "exponential", "variogram_parameters": [0.2, 8.0, 0.02], "coordinates_type": "geographic"}
    cases.append(run("geo", gc, gz, geo, "points"))
    cases.append(run("geo", gc, gz, geo, "grid"))
    cases.append(run("geo", gc, gz, geo, "grid", {"n_closest_points": 8, "backend": "loop"}))
    dc, dz = data["duplicated"]
    for kind in ["pinv", "pinvh"]:
        cases.append(run("duplicated", dc, dz, {"variogram_model": "linear", "variogram_parameters": [1.0, 0.1],
                                                 "pseudo_inv": True, "pseudo_inv_type": kind}, "points"))
    for name in ["meuse_logzinc", "pancake"]:
        c, z = data[name]
        for model in ["linear", "power", "gaussian", "spherical", "exponential", "hole-effect"]:
            for nlags, weight in [(6, False), (10, True)]:
                cases.append(run(name, c, z, {"variogram_model": model, "nlags": nlags, "weight": weight}, "points"))
    dump_json("kriging/ordinary.json", {
        "name": "kriging_ordinary",
        "description": "pykrige.OrdinaryKriging predictions, variances and automatic variograms",
        "meta": meta(),
        "datasets": {k: {"coordinates": c.tolist(), "values": z.tolist()} for k, (c, z) in data.items()},
        "cases": cases,
    })
    print(len(cases), "cases")


if __name__ == "__main__":
    main()
