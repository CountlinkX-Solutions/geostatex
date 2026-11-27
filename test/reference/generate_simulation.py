"""Generate GSTools simple kriging and simulation fixtures.

Writes `fixtures/kriging/simple.json` and `fixtures/simulation/srf.json`. Run from the
repository root (under a memory cap):

    prlimit --as=3221225472 .venv/bin/python -I test/reference/generate_simulation.py

Geostatex's (PyKrige) models map onto GSTools' with `rescale=1`: exponential
`len_scale = range / 3`, gaussian `len_scale = 4/7 range`, spherical `len_scale = range`,
`var = psill`; anisotropy `anis = 1 / anisotropy_scaling`, the angle in radians.

- Simple kriging (`gstools.krige.Simple`): a seeded synthetic 2D set (40 points in a 100 x 100
  square) and a 3D set, every model with a sill, points (two of them data points) and grids,
  exact or not, inverse or pseudo-inverses, a mean, anisotropy, no nugget.
- RNG: the first seeds of `MasterRNG` and draws of `RandomState(seed)` (normal, uniform).
- SRF (randomisation method): 2D gaussian and exponential (the models GSTools samples by
  inversion in 2D), with and without nugget, a mean, anisotropy, points and grids; the modes
  (`z_1`, `z_2`, wave vectors) of some cases.
- CondSRF: conditioned by simple and ordinary kriging of the 2D set.

Grids are stored with rows of y (Geostatex's layout); GSTools evaluates them x-major.
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gstools as gs  # noqa: E402
import numpy as np  # noqa: E402
from gstools.random.tools import MasterRNG  # noqa: E402

from common import dump_json, meta  # noqa: E402

warnings.simplefilter("ignore")

MODELS = {"exponential": gs.Exponential, "gaussian": gs.Gaussian, "spherical": gs.Spherical}


def gs_model(name, params, dim=2, scaling=1.0, angle=0.0):
    psill, rng, nugget = params
    length = {"exponential": rng / 3.0, "gaussian": rng * 4.0 / 7.0, "spherical": rng}[name]
    return MODELS[name](
        dim=dim, var=psill, len_scale=length, nugget=nugget, rescale=1.0,
        anis=1.0 / scaling, angles=np.deg2rad(angle),
    )


def rows_of_y(field, shape):
    """A field GSTools evaluated on an (x, y[, z]) mesh, as Geostatex's rows of y (z slowest)."""
    return np.asarray(field).reshape(shape).T.ravel()


def data_sets():
    rng = np.random.default_rng(13)
    x, y = rng.uniform(0, 100, 40), rng.uniform(0, 100, 40)
    v = np.sin(x / 20) + np.cos(y / 25) + rng.normal(0, 0.1, 40)
    px, py = np.concatenate([rng.uniform(0, 100, 10), x[:2]]), np.concatenate([rng.uniform(0, 100, 10), y[:2]])
    x3, y3, z3 = rng.uniform(0, 100, 40), rng.uniform(0, 100, 40), rng.uniform(0, 20, 40)
    v3 = np.sin(x3 / 20) + np.cos(y3 / 25) + 0.05 * z3 + rng.normal(0, 0.1, 40)
    p3 = [np.concatenate([rng.uniform(0, 100, 8), x3[:2]]), np.concatenate([rng.uniform(0, 100, 8), y3[:2]]),
          np.concatenate([rng.uniform(0, 20, 8), z3[:2]])]
    return {
        "d2": {"x": x, "y": y, "values": v, "px": px, "py": py,
               "gx": np.linspace(5, 95, 6), "gy": np.linspace(10, 90, 5)},
        "d3": {"x": x3, "y": y3, "z": z3, "values": v3, "px": p3[0], "py": p3[1], "pz": p3[2],
               "gx": np.linspace(5, 95, 4), "gy": np.linspace(10, 90, 3), "gz": np.linspace(2, 18, 2)},
    }


def simple_cases(d):
    cases = []

    def run(name, model, params, *, dim=2, style="points", mean=0.3, exact=True, pinv=None,
            scaling=1.0, angle=0.0):
        m = gs_model(model, params, dim, scaling, angle)
        s = d["d2"] if dim == 2 else d["d3"]
        axes = ["x", "y"] if dim == 2 else ["x", "y", "z"]
        k = gs.krige.Simple(m, [s[a] for a in axes], s["values"], mean=mean, exact=exact,
                            pseudo_inv=pinv is not None, pseudo_inv_type=pinv or "pinv")
        if style == "points":
            where = {a: s["p" + a] for a in axes}
            z, var = k([where[a] for a in axes], return_var=True)
        else:
            where = {a: s["g" + a] for a in axes}
            shape = tuple(len(where[a]) for a in axes)
            z, var = k.structured([where[a] for a in axes], return_var=True)
            z, var = rows_of_y(z, shape), rows_of_y(var, shape)
        cases.append({"name": name, "dim": dim, "variogram_model": model, "parameters": params,
                      "mean": mean, "exact_values": exact, "pseudo_inv": pinv,
                      "anisotropy_scaling": scaling, "anisotropy_angle": angle,
                      "style": style, "where": where, "z": z, "variance": var})

    for model in MODELS:
        for style in ["points", "grid"]:
            run(f"{model}_{style}", model, [0.8, 40.0, 0.05], style=style)
    exp = ("exponential", [0.8, 40.0, 0.05])
    run("not_exact", *exp, exact=False)
    run("pinv", *exp, pinv="pinv")
    run("pinvh", *exp, pinv="pinvh")
    run("no_nugget", "gaussian", [0.8, 40.0, 0.0])
    run("zero_mean", *exp, mean=0.0)
    run("anisotropy", *exp, scaling=2.0, angle=30.0)
    run("anisotropy_grid", "spherical", [0.8, 50.0, 0.05], style="grid", scaling=0.5, angle=-20.0)
    run("3d_points", "exponential", [0.8, 40.0, 0.05], dim=3)
    run("3d_grid", "gaussian", [0.8, 40.0, 0.05], dim=3, style="grid")
    return cases


def rng_section():
    out = []
    for seed in [0, 19, 4294967295]:
        master = MasterRNG(seed)
        first = int(master())
        out.append({
            "seed": seed,
            "master": [first] + [int(master()) for _ in range(7)],
            "normal": np.random.RandomState(seed).normal(size=7),
            "uniform": np.random.RandomState(seed).uniform(0.0, 2 * np.pi, 5),
            "rand": np.random.RandomState(seed).uniform(size=5),
        })
    return out


def srf_cases(d):
    s = d["d2"]
    shape = (len(s["gx"]), len(s["gy"]))
    cases = []

    def run(name, model, params, *, seed, mode_no=100, mean=0.0, scaling=1.0, angle=0.0,
            style="points", modes=False):
        m = gs_model(model, params, 2, scaling, angle)
        srf = gs.SRF(m, mean=mean, mode_no=mode_no, seed=seed)
        if style == "points":
            field = srf((s["px"], s["py"]))
            where = {"x": s["px"], "y": s["py"]}
        else:
            field = rows_of_y(srf.structured((s["gx"], s["gy"])), shape)
            where = {"x": s["gx"], "y": s["gy"]}
        case = {"name": name, "variogram_model": model, "parameters": params, "seed": seed,
                "mode_no": mode_no, "mean": mean, "anisotropy_scaling": scaling,
                "anisotropy_angle": angle, "style": style, "where": where, "field": field}
        if modes:
            gen = srf.generator
            case["modes"] = {"z_1": gen._z_1, "z_2": gen._z_2, "k": gen._cov_sample}
        cases.append(case)

    run("exponential", "exponential", [1.0, 30.0, 0.0], seed=19, modes=True)
    run("gaussian", "gaussian", [1.0, 30.0, 0.0], seed=19, modes=True)
    run("exponential_grid", "exponential", [1.0, 30.0, 0.0], seed=20, style="grid")
    run("gaussian_grid_nugget", "gaussian", [0.8, 30.0, 0.2], seed=21, style="grid", mean=1.5)
    run("nugget_mean", "exponential", [0.8, 30.0, 0.2], seed=22, mean=-0.5)
    run("anisotropy", "gaussian", [1.0, 30.0, 0.0], seed=23, scaling=2.0, angle=30.0)
    run("modes_1000", "exponential", [1.0, 30.0, 0.1], seed=24, mode_no=1000)
    return cases


def cond_cases(d):
    s = d["d2"]
    shape = (len(s["gx"]), len(s["gy"]))
    cases = []

    def run(name, kind, model, params, *, seed, mean=0.3, style="points", mode_no=100):
        m = gs_model(model, params)
        cond = [s["x"], s["y"]]
        if kind == "simple":
            k = gs.krige.Simple(m, cond, s["values"], mean=mean, exact=True, pseudo_inv=False)
        else:
            k = gs.krige.Ordinary(m, cond, s["values"], exact=True, pseudo_inv=False)
        srf = gs.CondSRF(k, mode_no=mode_no, seed=seed)
        if style == "points":
            field = srf((s["px"], s["py"]))
            where = {"x": s["px"], "y": s["py"]}
        else:
            field = rows_of_y(srf.structured((s["gx"], s["gy"])), shape)
            where = {"x": s["gx"], "y": s["gy"]}
        cases.append({"name": name, "kind": kind, "variogram_model": model, "parameters": params,
                      "mean": mean if kind == "simple" else None, "seed": seed, "mode_no": mode_no,
                      "style": style, "where": where, "field": field})

    run("simple", "simple", "exponential", [0.8, 40.0, 0.0], seed=31)
    run("simple_nugget", "simple", "gaussian", [0.8, 40.0, 0.05], seed=32)
    run("simple_grid", "simple", "exponential", [0.8, 40.0, 0.05], seed=33, style="grid")
    run("ordinary", "ordinary", "exponential", [0.8, 40.0, 0.05], seed=34)
    run("ordinary_grid", "ordinary", "gaussian", [0.8, 40.0, 0.0], seed=35, style="grid")
    return cases


def main():
    d = data_sets()
    simple = simple_cases(d)
    dump_json("kriging/simple.json", {
        "name": "kriging_simple",
        "description": "gstools.krige.Simple (PyKrige models mapped with rescale=1)",
        "meta": meta(),
        "data": {k: v for k, v in d.items()},
        "cases": simple,
    })
    srf, cond = srf_cases(d), cond_cases(d)
    dump_json("simulation/srf.json", {
        "name": "simulation_srf",
        "description": "gstools RNG, SRF (RandMeth) and CondSRF",
        "meta": meta(),
        "data": d["d2"],
        "rng": rng_section(),
        "srf": srf,
        "cond_srf": cond,
    })
    print(len(simple), "simple cases,", len(srf), "srf cases,", len(cond), "cond_srf cases")


if __name__ == "__main__":
    main()
