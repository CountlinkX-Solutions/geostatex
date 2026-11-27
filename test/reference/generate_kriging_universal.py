"""Generate PyKrige universal kriging fixtures (`fixtures/kriging/universal.json`).

Run from the repository root (under a memory cap):

    prlimit --as=3221225472 .venv/bin/python -I test/reference/generate_kriging_universal.py

Meuse log zinc (metres) with every drift: regional linear, point-log (two "wells"), an external
drift grid (bilinear), specified drifts and functional drifts (the functions are written in the
fixture so the Elixir test can rebuild them), alone and combined, with and without anisotropy,
explicit and automatic variograms, points and grids.
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import skgstat as skg  # noqa: E402
from pykrige.uk import UniversalKriging  # noqa: E402

from common import dump_json, meta  # noqa: E402

warnings.simplefilter("ignore")

# Functional drifts, also implemented in the Elixir test with the same names.
FUNCTIONS = {
    "xy": lambda x, y: (x - 179000.0) * (y - 331000.0) / 1.0e6,
    "dist_corner": lambda x, y: np.sqrt((x - 178500.0) ** 2 + (y - 333600.0) ** 2) / 1000.0,
}


def external_grid():
    gx = np.linspace(178500.0, 181600.0, 14)
    gy = np.linspace(329700.0, 333700.0, 17)
    gz = np.array([[0.002 * (x - 178000) + 0.001 * (y - 329000) + np.sin(x / 400.0) for x in gx] for y in gy])
    return gx, gy, gz


def main():
    c, z = skg.data.meuse(variable="zinc").get("sample")
    c = np.asarray(c, float)
    z = np.log(np.asarray(z, float).ravel())
    x, y = c[:, 0], c[:, 1]
    rng = np.random.default_rng(5)
    px = np.concatenate([rng.uniform(x.min(), x.max(), 12), x[:3]])
    py = np.concatenate([rng.uniform(y.min(), y.max(), 12), y[:3]])
    gx = np.linspace(x.min(), x.max(), 8)
    gy = np.linspace(y.min(), y.max(), 6)
    ex, ey, ez = external_grid()
    wells = [[179500.0, 331500.0, 1.0], [180500.0, 332500.0, 0.5]]
    spec_data = [x / 1000.0, (y - y.mean()) / 1000.0]

    def spec_at(xs, ys):
        return [np.asarray(xs) / 1000.0, (np.asarray(ys) - y.mean()) / 1000.0]

    variants = [
        ("regional", {"drift_terms": ["regional_linear"]}),
        ("point_log", {"drift_terms": ["point_log"], "point_drift": wells}),
        ("external", {"drift_terms": ["external_Z"], "external_drift": ez, "external_drift_x": ex, "external_drift_y": ey}),
        ("specified", {"drift_terms": ["specified"], "specified_drift": spec_data}),
        ("functional", {"drift_terms": ["functional"], "functional_drift": [FUNCTIONS["xy"], FUNCTIONS["dist_corner"]]}),
        ("combined", {"drift_terms": ["regional_linear", "point_log", "external_Z"], "point_drift": wells,
                      "external_drift": ez, "external_drift_x": ex, "external_drift_y": ey}),
    ]
    vario = {"variogram_model": "spherical", "variogram_parameters": {"psill": 0.6, "range": 1100.0, "nugget": 0.05}}
    cases = []
    for name, drift in variants:
        for aniso in [{}, {"anisotropy_scaling": 1.5, "anisotropy_angle": 20.0}]:
            if aniso and name in ("external", "specified"):
                continue
            uk = UniversalKriging(x, y, z, **vario, **aniso, **drift)
            for style in ["points", "grid"]:
                xs, ys = (px, py) if style == "points" else (gx, gy)
                kw = {}
                if "specified" in drift["drift_terms"]:
                    if style == "points":
                        kw["specified_drift_arrays"] = spec_at(px, py)
                    else:
                        mx, my = np.meshgrid(gx, gy)
                        kw["specified_drift_arrays"] = spec_at(mx, my)
                zv, ss = uk.execute(style, xs, ys, **kw)
                cases.append({"name": name, "anisotropy": aniso, "style": style,
                              "where": {"x": xs, "y": ys}, "z": np.asarray(zv).ravel(), "sigmasq": np.asarray(ss).ravel()})
    auto = UniversalKriging(x, y, z, variogram_model="exponential", nlags=8, drift_terms=["regional_linear"])
    zv, ss = auto.execute("points", px, py)
    dump_json("kriging/universal.json", {
        "name": "kriging_universal",
        "description": "pykrige.UniversalKriging with every drift",
        "meta": meta(),
        "data": {"coordinates": c.tolist(), "values": z.tolist()},
        "variogram": vario,
        "wells": wells,
        "external": {"x": ex, "y": ey, "z": ez},
        "y_mean": float(y.mean()),
        "cases": cases,
        "auto": {"variogram_model_parameters": list(auto.variogram_model_parameters), "lags": auto.lags,
                 "semivariance": auto.semivariance, "where": {"x": px, "y": py},
                 "z": zv, "sigmasq": ss},
    })
    print(len(cases), "cases")


if __name__ == "__main__":
    main()
