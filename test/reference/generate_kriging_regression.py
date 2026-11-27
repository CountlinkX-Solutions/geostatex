"""Generate PyKrige regression-kriging fixtures (`fixtures/kriging/regression.json`).

Run from the repository root (under a memory cap):

    prlimit --as=3221225472 .venv/bin/python -I test/reference/generate_kriging_regression.py

`pykrige.rk.RegressionKriging` fits a scikit-learn regression on the predictors, then krige the
residuals with `pykrige.compat.Krige` (ordinary kriging with a moving window of
`n_closest_points` and the loop backend, or universal kriging without one); a prediction is the
kriged residual plus the regression. The regression here is sklearn's `LinearRegression`
(ordinary least squares with an intercept).

Data: the first 340 of the 2,000 King County sales in `notebooks/data/king_county` (OpenML
42092, CC0): 300 to fit, 40 to predict. Target: log price; predictors: log living area, grade,
bathrooms, year built; coordinates: longitude / latitude projected to kilometres around the
sample's centre.
"""

import contextlib
import csv
import io
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from pykrige.rk import RegressionKriging  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402

from common import dump_json, meta  # noqa: E402

warnings.simplefilter("ignore")

ROOT = Path(__file__).resolve().parents[2]


def load():
    with open(ROOT / "notebooks/data/king_county/sales_sample.csv") as f:
        rows = list(csv.DictReader(f))[:340]
    lat = np.array([float(r["lat"]) for r in rows])
    lon = np.array([float(r["long"]) for r in rows])
    lat0, lon0 = lat.mean(), lon.mean()
    x = (lon - lon0) * np.cos(np.radians(lat0)) * 111.32
    y = (lat - lat0) * 110.57
    p = np.column_stack([
        np.log([float(r["sqft_living"]) for r in rows]),
        [float(r["grade"]) for r in rows],
        [float(r["bathrooms"]) for r in rows],
        [float(r["yr_built"]) for r in rows],
    ])
    target = np.log([float(r["price"]) for r in rows])
    return np.column_stack([x, y]), p, target


def run(c, p, z, case):
    fit, new = slice(0, 300), slice(300, 340)
    rk = RegressionKriging(regression_model=LinearRegression(), **case)
    with contextlib.redirect_stdout(io.StringIO()):
        rk.fit(p[fit], c[fit], z[fit])
        prediction = rk.predict(p[new], c[new])
        residual = rk.krige_residual(c[new])
        score = rk.score(p[new], c[new], z[new])
    model = rk.krige.model
    return {
        "params": case,
        "coef": rk.regression_model.coef_,
        "intercept": float(rk.regression_model.intercept_),
        "variogram_model_parameters": list(model.variogram_model_parameters),
        "lags": model.lags,
        "semivariance": model.semivariance,
        "prediction": prediction,
        "residual": residual,
        "score": float(score),
    }


def main():
    c, p, z = load()
    cases = [
        {"method": "ordinary"},
        {"method": "ordinary", "variogram_model": "exponential", "nlags": 10},
        {"method": "ordinary", "variogram_model": "spherical", "n_closest_points": 20,
         "variogram_parameters": {"psill": 0.02, "range": 6.0, "nugget": 0.015}},
        {"method": "universal", "variogram_model": "linear", "drift_terms": ["regional_linear"]},
    ]
    dump_json("kriging/regression.json", {
        "name": "kriging_regression",
        "description": "pykrige.rk.RegressionKriging with sklearn LinearRegression",
        "meta": meta(),
        "coordinates": c.tolist(),
        "predictors": p.tolist(),
        "values": z.tolist(),
        "fit_rows": 300,
        "cases": [run(c, p, z, case) for case in cases],
    })
    print(len(cases), "cases")


if __name__ == "__main__":
    main()
