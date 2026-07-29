"""Benchmarks against upstream shap. Run only through ``pixi run bench``."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np
import shap
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"),
)
import mojo_shap


def timeit(fn, repeat=4):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def main():
    rng = np.random.default_rng(20)
    X = rng.normal(size=(12_000, 20))
    y = X[:, 0] * X[:, 1] - 2 * X[:, 2] + np.sin(X[:, 3])
    forest = RandomForestRegressor(
        n_estimators=100, max_depth=10, random_state=0, n_jobs=1
    ).fit(X, y)
    query = np.ascontiguousarray(X[:1000])

    tree_ours = mojo_shap.TreeExplainer(forest)
    tree_upstream = shap.TreeExplainer(forest)

    background = np.ascontiguousarray(X[:100])
    kernel_query = np.ascontiguousarray(X[100:110, :10])
    coefficient = rng.normal(size=10)

    def model(z):
        return np.tanh(z @ coefficient)

    kernel_ours = mojo_shap.KernelExplainer(model, background[:, :10])
    kernel_upstream = shap.KernelExplainer(model, background[:, :10])

    cases = [
        (
            "TreeSHAP RF 100 trees depth 10, 1k rows",
            lambda: tree_ours.shap_values(query),
            lambda: tree_upstream.shap_values(query),
        ),
        (
            "Kernel SHAP exact, 10 rows x 10 features",
            lambda: kernel_ours.shap_values(
                kernel_query, nsamples=1022, l1_reg=0
            ),
            lambda: kernel_upstream.shap_values(
                kernel_query, nsamples=1022, l1_reg=0, silent=True
            ),
        ),
    ]

    print(f"Machine: {cpu_name()}")
    print()
    print("| case | mojo-shap | shap | result |")
    print("| --- | ---: | ---: | ---: |")
    for name, ours, upstream in cases:
        ours()
        upstream()
        a = timeit(ours)
        b = timeit(upstream)
        ratio = b / a
        result = (
            f"{ratio:.2f}x faster"
            if ratio >= 1
            else f"{1 / ratio:.2f}x slower"
        )
        print(f"| {name} | {a * 1000:.1f} ms | {b * 1000:.1f} ms | {result} |")


if __name__ == "__main__":
    main()
