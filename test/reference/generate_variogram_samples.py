"""Generate pair-sampling fixtures (`fixtures/variogram/samples.json`).

Run from the repository root (under a memory cap):

    prlimit --as=3221225472 .venv/bin/python -I test/reference/generate_variogram_samples.py

`skgstat.Variogram(samples=..., binning_random_state=seed)` measures the distances between two
random subsets of the points (`ProbabalisticMetricSpace`): `lidx` then `ridx`, each drawn with
`RandomState(MT19937(SeedSequence(seed))).choice(n, k, replace=False)`, `k` an int or
`int(samples * n)`. The variogram keeps the pairs `(l, r)` with `r < l` and a non-zero
distance; this script checks that rule against scikit-gstat's own distances and differences.

`rng` records numpy's draws directly (SeedSequence words, raw MT19937 outputs, two successive
choices); `cases` record the variograms. Datasets as in `experimental.json`: Meuse lead, the
pancake sample and a seeded synthetic set with duplicate points.
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import skgstat as skg  # noqa: E402

from common import dump_json, meta  # noqa: E402

warnings.simplefilter("ignore")

SEEDS = [0, 1, 42, 123456789, 2**32 - 1, 2**32 + 5, 2**63 + 17, 2**64 - 1]


def datasets():
    out = {}
    c, z = skg.data.meuse(variable="lead").get("sample")
    out["meuse_lead"] = (np.asarray(c, float), np.asarray(z, float).ravel())
    c, z = skg.data.pancake(N=150, seed=42).get("sample")
    out["pancake"] = (np.asarray(c, float), np.asarray(z, float).ravel())
    rng = np.random.default_rng(7)
    c = rng.uniform(0, 100, size=(60, 2))
    c[10] = c[3]
    c[25] = c[3]
    z = 5 + 0.05 * c[:, 0] + rng.normal(0, 1, 60)
    out["synthetic_dups"] = (c, z)
    return out


def rng_case(seed, n, k):
    words = np.random.SeedSequence(seed).generate_state(8, np.uint32)
    raw = np.random.MT19937(np.random.SeedSequence(seed)).random_raw(700)
    rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(seed)))
    first = rs.choice(n, size=k, replace=False)
    second = rs.choice(n, size=k, replace=False)
    return {"seed": str(seed), "n": n, "k": k, "seed_words": words.tolist(),
            "raw": raw.tolist(), "first": first.tolist(), "second": second.tolist()}


def check_pair_rule(v, coords, values):
    lidx, ridx = v._X.lidx, v._X.ridx
    pairs = sorted({(int(l), int(r)) for l in lidx for r in ridx if r < l})
    kept = [(l, r) for l, r in pairs if np.linalg.norm(coords[l] - coords[r]) > 0]
    d = np.array([np.linalg.norm(coords[l] - coords[r]) for l, r in kept])
    diffs = np.array([abs(values[l] - values[r]) for l, r in kept])
    assert len(d) == len(v.distance), (len(d), len(v.distance))
    assert np.allclose(np.sort(d), np.sort(v.distance), rtol=0, atol=1e-9)
    assert np.allclose(np.sort(diffs), np.sort(v._diff), rtol=0, atol=1e-9)


def record(data, name, params):
    coords, values = data[name]
    v = skg.Variogram(coords, values, fit_method=None, **params)
    check_pair_rule(v, coords, values)
    return {
        "dataset": name,
        "params": {k: (str(val) if k == "binning_random_state" else val)
                   for k, val in params.items()},
        "lidx": v._X.lidx.tolist(),
        "ridx": v._X.ridx.tolist(),
        "pairs": int(len(v.distance)),
        "bins": v.bins,
        "experimental": v.experimental,
        "bin_count": v.bin_count,
        "n_lags": int(v.n_lags),
        "maxlag": None if v.maxlag is None else float(v.maxlag),
    }


def main():
    rng = [rng_case(seed, n, k) for seed in SEEDS for n, k in [(10, 3), (155, 77), (60, 60)]]
    data = datasets()
    cases = []
    for name in data:
        for samples, seed in [(40, 0), (0.5, 42), (60, 2**63 + 17)]:
            cases.append(record(data, name, {"samples": samples, "binning_random_state": seed,
                                             "n_lags": 8}))
    for params in [
        {"estimator": "cressie"}, {"estimator": "dowd"}, {"estimator": "percentile"},
        {"maxlag": 0.5}, {"maxlag": "median"}, {"bin_func": "uniform"}, {"bin_func": "fd"},
    ]:
        cases.append(record(data, "meuse_lead", {"samples": 0.6, "binning_random_state": 7,
                                                 "n_lags": 6, **params}))
    dump_json("variogram/samples.json", {
        "name": "variogram_samples",
        "description": "skgstat.Variogram with samples (ProbabalisticMetricSpace) and numpy draws",
        "meta": meta(),
        "datasets": {k: {"coordinates": c.tolist(), "values": z.tolist()}
                     for k, (c, z) in data.items()},
        "rng": rng,
        "cases": cases,
    })
    print(len(rng), "rng cases,", len(cases), "variogram cases")


if __name__ == "__main__":
    main()
