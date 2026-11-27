"""Generate GSTools covariance model fixtures (`fixtures/models/gstools.json`).

Run from the repository root (under a memory cap):

    prlimit --as=3221225472 .venv/bin/python -I test/reference/generate_gstools_models.py

The GSTools 1.7.0 models Geostatex had not ported (Integral, Rational, Circular, SuperSpherical
and HyperSpherical, JBessel, and the truncated power laws with `len_low = 0`), with `rescale=1`
so that Geostatex's `range` is GSTools' `len_scale`: the variogram `nugget + var * (1 - cor(r /
len_scale))` at distances up to four length scales, for several shape parameters; and ordinary
kriging (`gstools.krige.Ordinary`, exact, no pseudo-inverse) of a seeded 2D set with some of them.
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gstools as gs  # noqa: E402
import numpy as np  # noqa: E402

from common import dump_json, meta  # noqa: E402

warnings.simplefilter("ignore")

# Geostatex name -> (GSTools class, its shape arguments).
MODELS = {
    "integral": (gs.Integral, ["nu"]),
    "rational": (gs.Rational, ["alpha"]),
    "circular": (gs.Circular, []),
    "superspherical": (gs.SuperSpherical, ["nu"]),
    "jbessel": (gs.JBessel, ["nu"]),
    "tpl_gaussian": (gs.TPLGaussian, ["hurst"]),
    "tpl_exponential": (gs.TPLExponential, ["hurst"]),
    "tpl_stable": (gs.TPLStable, ["hurst", "alpha"]),
    "tpl_simple": (gs.TPLSimple, ["nu"]),
}

SHAPES = {
    "integral": [{"nu": 0.5}, {"nu": 1.0}, {"nu": 2.0}, {"nu": 7.3}],
    "rational": [{"alpha": 0.5}, {"alpha": 1.0}, {"alpha": 4.5}],
    "circular": [{}],
    "superspherical": [{"nu": 0.5}, {"nu": 1.0}, {"nu": 3.7}],
    "jbessel": [{"nu": 0.5}, {"nu": 1.0}, {"nu": 2.5}],
    "tpl_gaussian": [{"hurst": 0.25}, {"hurst": 0.5}, {"hurst": 0.9}],
    "tpl_exponential": [{"hurst": 0.25}, {"hurst": 0.5}, {"hurst": 0.9}],
    "tpl_stable": [{"hurst": 0.5, "alpha": 1.5}, {"hurst": 0.3, "alpha": 0.8}],
    "tpl_simple": [{"nu": 1.5}, {"nu": 2.0}, {"nu": 4.2}],
}


def model(name, var, len_scale, nugget, shape, dim=2):
    cls, _ = MODELS[name]
    kw = dict(shape)
    if name.startswith("tpl_") and name != "tpl_simple":
        kw["len_low"] = 0.0
    return cls(dim=dim, var=var, len_scale=len_scale, nugget=nugget, rescale=1.0, **kw)


def main():
    distances = np.concatenate([[0.0, 1e-9, 0.05, 0.3], np.linspace(0.5, 40.0, 60)])
    values = []
    for name, shapes in SHAPES.items():
        for shape in shapes:
            m = model(name, 1.3, 10.0, 0.2, shape)
            values.append({"model": name, "shape": shape, "var": 1.3, "len_scale": 10.0,
                           "nugget": 0.2, "variogram": m.variogram(distances)})
    # HyperSpherical is SuperSpherical with nu = (dim - 1) / 2.
    for dim in [2, 3]:
        h = gs.HyperSpherical(dim=dim, var=1.3, len_scale=10.0, nugget=0.2, rescale=1.0)
        values.append({"model": "hyperspherical", "dim": dim, "shape": {"nu": (dim - 1) / 2},
                       "var": 1.3, "len_scale": 10.0, "nugget": 0.2, "variogram": h.variogram(distances)})

    rng = np.random.default_rng(17)
    x, y = rng.uniform(0, 100, 40), rng.uniform(0, 100, 40)
    v = np.sin(x / 20) + np.cos(y / 25) + rng.normal(0, 0.1, 40)
    px, py = np.concatenate([rng.uniform(0, 100, 10), x[:2]]), np.concatenate([rng.uniform(0, 100, 10), y[:2]])
    kriging = []
    for name, shape in [("rational", {"alpha": 1.0}), ("superspherical", {"nu": 1.0}),
                        ("jbessel", {"nu": 1.0}), ("tpl_gaussian", {"hurst": 0.5}),
                        ("integral", {"nu": 2.0}), ("circular", {})]:
        m = model(name, 0.8, 30.0, 0.05, shape)
        k = gs.krige.Ordinary(m, [x, y], v, exact=True, pseudo_inv=False)
        z, var = k([px, py], return_var=True)
        kriging.append({"model": name, "shape": shape, "var": 0.8, "len_scale": 30.0, "nugget": 0.05,
                        "z": z, "variance": var})
    dump_json("models/gstools.json", {
        "name": "gstools_models",
        "description": "GSTools covariance models (rescale=1): variograms and ordinary kriging",
        "meta": meta(),
        "distances": distances,
        "values": values,
        "data": {"x": x, "y": y, "values": v, "px": px, "py": py},
        "kriging": kriging,
    })
    print(len(values), "model curves,", len(kriging), "kriging cases")


if __name__ == "__main__":
    main()
