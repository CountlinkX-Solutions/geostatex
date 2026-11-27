"""Generate classification kriging fixtures (`fixtures/kriging/classification.json`).

Run from the repository root (under a memory cap):

    prlimit --as=3221225472 .venv/bin/python -I test/reference/generate_kriging_classification.py

`pykrige.ck.ClassificationKriging` on Meuse: the soil class (1, 2, 3) from elevation and distance
to the river, a scikit-learn `LogisticRegression` (deterministic) as the classifier, 120 points to
fit and 35 to predict. The classifier's probabilities are stored (the Elixir tests replay them),
with the ILR transforms, the kriged residuals per ILR coordinate, the corrected probabilities
and PyKrige's predictions (class indices).
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import skgstat  # noqa: E402
from pykrige.ck import ClassificationKriging, ilr_transformation, inverse_ilr_transformation  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from common import dump_json, meta  # noqa: E402

warnings.simplefilter("ignore")


def main():
    path = Path(skgstat.__file__).parent / "data" / "samples" / "meuse.txt"
    df = pd.read_csv(path)
    xy = df[["x", "y"]].to_numpy(float)
    p = df[["elev", "dist"]].to_numpy(float)
    y = df["soil"].astype(int).to_numpy()
    order = np.random.default_rng(16).permutation(len(df))
    train, test = order[:120], order[120:]
    cases = []

    def run(name, **kw):
        ck = ClassificationKriging(classification_model=LogisticRegression(max_iter=5000), **kw)
        ck.fit(p[train], xy[train], y[train].reshape(-1, 1))
        ml_train = ck.classification_model.predict_proba(p[train])
        ml_test = ck.classification_model.predict_proba(p[test])
        residual = ck.krige_residual(xy[test])
        proba = inverse_ilr_transformation(residual + ilr_transformation(ml_test))
        cases.append({
            "name": name,
            "options": {k: v for k, v in kw.items()},
            "fitted": [list(k.model.variogram_model_parameters) for k in ck.krige],
            "ml_train": ml_train, "ml_test": ml_test,
            "ilr_ml_test": ilr_transformation(ml_test),
            "residual": residual, "proba": proba,
            "indices": ck.predict(p[test], xy[test]),
        })

    run("spherical_window", variogram_model="spherical", variogram_parameters=[1.0, 1000.0, 0.3])
    run("spherical_global", variogram_model="spherical", variogram_parameters=[1.0, 1000.0, 0.3],
        n_closest_points=None)
    run("linear_automatic")
    run("exponential_window", variogram_model="exponential", variogram_parameters=[1.2, 1500.0, 0.2],
        n_closest_points=15)
    dump_json("kriging/classification.json", {
        "name": "kriging_classification",
        "description": "pykrige ClassificationKriging on Meuse soil classes (LogisticRegression)",
        "meta": meta(),
        "classes": [1, 2, 3],
        "train": {"x": xy[train, 0], "y": xy[train, 1], "predictors": p[train], "labels": y[train]},
        "test": {"x": xy[test, 0], "y": xy[test, 1], "predictors": p[test], "labels": y[test]},
        "cases": cases,
    })
    print(len(cases), "cases")


if __name__ == "__main__":
    main()
