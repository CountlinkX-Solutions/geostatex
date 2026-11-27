"""Generate PyKrige model statistics and leave-one-out fixtures (`fixtures/kriging/validation.json`).

Run from the repository root (under a memory cap):

    prlimit --as=3221225472 .venv/bin/python -I test/reference/generate_kriging_validation.py

* `statistics`: PyKrige's `delta`, `sigma`, `epsilon`, `Q1`, `Q2` and `cR` (each point kriged from
  the points before it) of ordinary (`enable_statistics=True`), universal and 3D models.
* `loo`: leave-one-out kriging with PyKrige itself: the variogram of the model on all the data,
  then for every point a model on the other points with those parameters, executed at the point.

The datasets are those of `generate_kriging_ordinary.py`, plus a seeded 3D set.
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from pykrige.ok import OrdinaryKriging  # noqa: E402
from pykrige.ok3d import OrdinaryKriging3D  # noqa: E402
from pykrige.uk import UniversalKriging  # noqa: E402

from common import dump_json, meta  # noqa: E402
from generate_kriging_ordinary import datasets  # noqa: E402

warnings.simplefilter("ignore")


def floats(a):
    return [float(v) for v in np.asarray(a).ravel()]


def build(kind, c, z, init):
    if kind == "ordinary":
        return OrdinaryKriging(c[:, 0], c[:, 1], z, **init)
    if kind == "universal":
        return UniversalKriging(c[:, 0], c[:, 1], z, **init)
    return OrdinaryKriging3D(c[:, 0], c[:, 1], c[:, 2], z, **init)


def statistics(name, kind, c, z, init):
    k = build(kind, c, z, dict(init, enable_statistics=True) if kind == "ordinary" else init)
    return {"dataset": name, "kind": kind, "init": init, "params": floats(k.variogram_model_parameters),
            "delta": floats(k.delta), "sigma": floats(k.sigma), "epsilon": floats(k.epsilon),
            "Q1": float(k.Q1), "Q2": float(k.Q2), "cR": float(k.cR)}


def fixed(model, params):
    # PyKrige reads a list as [sill, range, nugget] (the full sill) for these models; the
    # dictionary form takes its own psill.
    if model in ("linear", "power"):
        return params
    return {"psill": params[0], "range": params[1], "nugget": params[2]}


def loo(name, kind, c, z, init, execute=None):
    execute = execute or {}
    full = build(kind, c, z, init)
    params = floats(full.variogram_model_parameters)
    pred, var = [], []
    for i in range(len(z)):
        keep = np.arange(len(z)) != i
        init_i = dict(init, variogram_parameters=fixed(init["variogram_model"], params))
        ex = dict(execute)
        if "specified_drift" in init:
            init_i["specified_drift"] = [np.asarray(d)[keep] for d in init["specified_drift"]]
            ex["specified_drift_arrays"] = [np.asarray([d[i]]) for d in init["specified_drift"]]
        k = build(kind, c[keep], z[keep], init_i)
        if kind == "ordinary3d":
            p, s = k.execute("points", c[i:i + 1, 0], c[i:i + 1, 1], c[i:i + 1, 2], **ex)
        else:
            p, s = k.execute("points", c[i:i + 1, 0], c[i:i + 1, 1], **ex)
        pred.append(float(np.asarray(p).ravel()[0]))
        var.append(float(np.asarray(s).ravel()[0]))
    init_json = {k: (v if k != "specified_drift" else [floats(d) for d in v]) for k, v in init.items()}
    return {"dataset": name, "kind": kind, "init": init_json, "execute": execute, "params": params,
            "predictions": pred, "variances": var}


def main():
    data = datasets()
    rng = np.random.default_rng(17)
    c3 = rng.uniform(0, 10, (40, 3))
    data["cube"] = (c3, 1 + 0.3 * c3[:, 0] - 0.2 * c3[:, 2] + rng.normal(0, 0.3, 40))
    sph = {"variogram_model": "spherical"}
    stats = [
        statistics("meuse_logzinc", "ordinary", *data["meuse_logzinc"], sph),
        statistics("meuse_logzinc", "ordinary", *data["meuse_logzinc"], {"variogram_model": "exponential"}),
        statistics("meuse_logzinc", "ordinary", *data["meuse_logzinc"], {"variogram_model": "linear"}),
        statistics("meuse_logzinc", "ordinary", *data["meuse_logzinc"],
                   dict(sph, anisotropy_scaling=2.0, anisotropy_angle=30.0)),
        statistics("meuse_logzinc", "ordinary", *data["meuse_logzinc"], dict(sph, pseudo_inv=True)),
        statistics("pancake", "ordinary", *data["pancake"], {"variogram_model": "gaussian"}),
        statistics("geo", "ordinary", *data["geo"], dict(sph, coordinates_type="geographic")),
        statistics("duplicated", "ordinary", *data["duplicated"], {"variogram_model": "linear"}),
        statistics("meuse_logzinc", "universal", *data["meuse_logzinc"],
                   dict(sph, drift_terms=["regional_linear"])),
        statistics("cube", "ordinary3d", *data["cube"], {"variogram_model": "exponential"}),
    ]
    c, z = data["meuse_logzinc"]
    drift = [c[:, 0] / 1000.0]
    loos = [
        loo("meuse_logzinc", "ordinary", c, z, sph),
        loo("meuse_logzinc", "ordinary", c, z, sph, {"n_closest_points": 12, "backend": "loop"}),
        loo("meuse_logzinc", "ordinary", c, z, dict(sph, anisotropy_scaling=2.0, anisotropy_angle=30.0)),
        loo("pancake", "ordinary", *data["pancake"], {"variogram_model": "gaussian"}),
        loo("geo", "ordinary", *data["geo"], dict(sph, coordinates_type="geographic")),
        loo("meuse_logzinc", "universal", c, z, dict(sph, drift_terms=["regional_linear"])),
        loo("meuse_logzinc", "universal", c, z, dict(sph, drift_terms=["specified"], specified_drift=drift)),
        loo("cube", "ordinary3d", *data["cube"], {"variogram_model": "exponential"}),
    ]
    dump_json("kriging/validation.json", {
        "name": "kriging_validation",
        "description": "PyKrige statistics (Q1, Q2, cR) and leave-one-out kriging",
        "meta": meta(),
        "datasets": {k: {"coordinates": c.tolist(), "values": z.tolist()} for k, (c, z) in data.items()},
        "statistics": stats,
        "loo": loos,
    })
    print(len(stats), "statistics,", len(loos), "leave-one-out")


if __name__ == "__main__":
    main()
