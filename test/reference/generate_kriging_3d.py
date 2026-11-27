"""Generate PyKrige 3D kriging fixtures (`fixtures/kriging/kriging3d.json`).

Run from the repository root (under a memory cap):

    prlimit --as=3221225472 .venv/bin/python -I test/reference/generate_kriging_3d.py

A seeded synthetic 3D set (90 points in a 100 x 100 x 20 box, a smooth field plus noise):
OrdinaryKriging3D with every model, points / grid / masked, full 3D anisotropy, exact_values
false, a moving window, a pseudo-inverse and an automatic variogram; UniversalKriging3D with
regional linear, specified and functional drifts.
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from pykrige.ok3d import OrdinaryKriging3D  # noqa: E402
from pykrige.uk3d import UniversalKriging3D  # noqa: E402

from common import dump_json, meta  # noqa: E402

warnings.simplefilter("ignore")


def main():
    rng = np.random.default_rng(31)
    x, y, z = rng.uniform(0, 100, 90), rng.uniform(0, 100, 90), rng.uniform(0, 20, 90)
    v = np.sin(x / 15) + np.cos(y / 20) + 0.05 * z + rng.normal(0, 0.1, 90)
    px, py, pz = rng.uniform(0, 100, 10), rng.uniform(0, 100, 10), rng.uniform(0, 20, 10)
    px, py, pz = np.concatenate([px, x[:2]]), np.concatenate([py, y[:2]]), np.concatenate([pz, z[:2]])
    gx, gy, gz = np.linspace(5, 95, 5), np.linspace(10, 90, 4), np.linspace(2, 18, 3)
    mask = np.zeros((3, 4, 5), dtype=bool)
    mask[1, ::2, 1::2] = True
    explicit = {
        "linear": [0.02, 0.01], "power": [0.05, 1.2, 0.01], "gaussian": [0.8, 60.0, 0.01],
        "spherical": [0.8, 70.0, 0.01], "exponential": [0.8, 90.0, 0.01], "hole-effect": [0.8, 50.0, 0.01],
    }
    cases = []

    def run(name, init, style, cls=OrdinaryKriging3D, execute=None):
        k = cls(x, y, z, v, **init)
        execute = dict(execute or {})
        if style == "points":
            zv, ss = k.execute("points", px, py, pz, **execute)
            where = {"x": px, "y": py, "z": pz}
        else:
            if style == "masked":
                execute["mask"] = mask
            zv, ss = k.execute(style, gx, gy, gz, **execute)
            zv, ss = np.ma.filled(zv, np.nan), np.ma.filled(ss, np.nan)
            where = {"x": gx, "y": gy, "z": gz}
        init_json = {kk: (vv if not isinstance(vv, (list, tuple, np.ndarray)) else [list(a) if isinstance(a, np.ndarray) else a for a in vv]) for kk, vv in init.items() if kk not in ("functional_drift", "specified_drift")}
        cases.append({"name": name, "init": init_json, "style": style,
                      "execute": {kk: vv for kk, vv in execute.items() if kk not in ("mask", "specified_drift_arrays")},
                      "where": where, "variogram_model_parameters": list(k.variogram_model_parameters),
                      "lags": k.lags, "semivariance": k.semivariance,
                      "z": np.asarray(zv).ravel(), "sigmasq": np.asarray(ss).ravel()})

    for model, params in explicit.items():
        for style in ["points", "grid"]:
            run(model, {"variogram_model": model, "variogram_parameters": params}, style)
    sph = {"variogram_model": "spherical", "variogram_parameters": [0.8, 70.0, 0.01]}
    run("masked", sph, "masked")
    run("anisotropy", dict(sph, anisotropy_scaling_y=2.0, anisotropy_scaling_z=0.5, anisotropy_angle_x=10.0,
                           anisotropy_angle_y=20.0, anisotropy_angle_z=30.0), "points")
    run("not_exact", dict(sph, exact_values=False), "points")
    run("window", sph, "points", execute={"n_closest_points": 12, "backend": "loop"})
    run("pinv", dict(sph, pseudo_inv=True), "points")
    run("auto", {"variogram_model": "exponential", "nlags": 8}, "points")
    run("uk_regional", dict(sph, drift_terms=["regional_linear"]), "points", cls=UniversalKriging3D)
    run("uk_regional_aniso", dict(sph, drift_terms=["regional_linear"], anisotropy_scaling_z=2.0, anisotropy_angle_z=40.0),
        "grid", cls=UniversalKriging3D)
    run("uk_specified", dict(sph, drift_terms=["specified"], specified_drift=[z / 10.0]), "points", cls=UniversalKriging3D,
        execute={"specified_drift_arrays": [pz / 10.0]})
    run("uk_functional", dict(sph, drift_terms=["functional"], functional_drift=[lambda a, b, c: (a - 50.0) * (c - 10.0) / 100.0]),
        "points", cls=UniversalKriging3D)
    dump_json("kriging/kriging3d.json", {
        "name": "kriging_3d",
        "description": "pykrige OrdinaryKriging3D and UniversalKriging3D",
        "meta": meta(),
        "data": {"x": x, "y": y, "z": z, "values": v},
        "mask": mask.astype(int).tolist(),
        "cases": cases,
    })
    print(len(cases), "cases")


if __name__ == "__main__":
    main()
