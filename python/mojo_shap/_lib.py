"""ctypes bridge to the single Mojo compilation unit."""

from __future__ import annotations

import ctypes
import os
import subprocess
from collections.abc import Sequence

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJO_SHAP_LIB", os.path.join(ROOT, "dist", "libmojo-shap.so"))
I = ctypes.c_int64

_SIGNATURES = {
    "msh_tree_shap": ([I] * 20, I),
    "msh_make_synthetic": ([I] * 9, I),
}

_library: ctypes.CDLL | None = None


def build() -> str:
    if not os.path.exists(LIB):
        subprocess.run(
            ["bash", os.path.join(ROOT, "build", "build.sh")],
            cwd=ROOT,
            check=True,
        )
    return LIB


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_library, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _library


def f64(value) -> np.ndarray:
    source = np.asarray(value)
    if source.dtype.kind not in "biuf":
        raise TypeError("inputs must contain real numeric values")
    if source.dtype.kind in "iu":
        if np.any(source > 2**53) or (
            source.dtype.kind == "i" and np.any(source < -(2**53))
        ):
            raise ValueError(
                "integer input outside the exact float64 safety range would risk "
                "silent narrowing"
            )
    return np.ascontiguousarray(value, dtype=np.float64)


def i64(value) -> np.ndarray:
    return np.ascontiguousarray(value, dtype=np.int64)


def addr(value: np.ndarray) -> int:
    return int(value.ctypes.data)


def _array(
    value: np.ndarray,
    dtype: np.dtype,
    name: str,
    *,
    ndim: int = 1,
    writable: bool = False,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}")
    if value.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    if writable and not value.flags.writeable:
        raise ValueError(f"{name} must be writable")
    if value.size and not addr(value):
        raise ValueError(f"{name} has a null data pointer")
    return value


def _check_lengths(arrays: Sequence[tuple[np.ndarray, int, str]]) -> None:
    for value, minimum, name in arrays:
        if minimum < 0 or value.size < minimum:
            raise ValueError(f"{name} is shorter than required")


def tree_shap(
    x: np.ndarray,
    children_left: np.ndarray,
    children_right: np.ndarray,
    feature: np.ndarray,
    threshold: np.ndarray,
    value: np.ndarray,
    node_weight: np.ndarray,
    missing_left: np.ndarray,
    offsets: np.ndarray,
    scales: np.ndarray,
    n_trees: int,
    n_outputs: int,
    phi: np.ndarray,
    path_feature: np.ndarray,
    path_zero: np.ndarray,
    path_one: np.ndarray,
    path_weight: np.ndarray,
    path_stride: int,
) -> None:
    x = _array(x, np.dtype(np.float64), "x", ndim=2)
    phi = _array(phi, np.dtype(np.float64), "phi", ndim=3, writable=True)
    int_arrays = [
        _array(children_left, np.dtype(np.int64), "children_left"),
        _array(children_right, np.dtype(np.int64), "children_right"),
        _array(feature, np.dtype(np.int64), "feature"),
        _array(missing_left, np.dtype(np.int64), "missing_left"),
    ]
    float_arrays = [
        _array(threshold, np.dtype(np.float64), "threshold"),
        _array(node_weight, np.dtype(np.float64), "node_weight"),
    ]
    offsets = _array(offsets, np.dtype(np.int64), "offsets")
    scales = _array(scales, np.dtype(np.float64), "scales")
    value = _array(value, np.dtype(np.float64), "value", ndim=2)
    path_feature = _array(
        path_feature, np.dtype(np.int64), "path_feature", writable=True
    )
    scratch = [
        _array(path_zero, np.dtype(np.float64), "path_zero", writable=True),
        _array(path_one, np.dtype(np.float64), "path_one", writable=True),
        _array(path_weight, np.dtype(np.float64), "path_weight", writable=True),
    ]
    if n_trees <= 0 or n_outputs <= 0 or path_stride < 0:
        raise ValueError("tree dimensions must be positive")
    if phi.shape != (len(x), x.shape[1], n_outputs):
        raise ValueError("phi shape does not match x and n_outputs")
    node_count = len(children_left)
    _check_lengths(
        [(a, node_count, name) for a, name in zip(
            int_arrays[1:] + float_arrays,
            ["children_right", "feature", "missing_left", "threshold", "node_weight"],
        )]
        + [
            (offsets, n_trees, "offsets"),
            (scales, n_trees, "scales"),
        ]
    )
    if value.shape != (node_count, n_outputs):
        raise ValueError("value shape does not match nodes and outputs")
    scratch_size = len(path_feature)
    _check_lengths(
        [(a, scratch_size, name) for a, name in zip(
            scratch, ["path_zero", "path_one", "path_weight"]
        )]
    )
    status = lib().msh_tree_shap(
        addr(x), len(x), x.shape[1],
        addr(children_left), addr(children_right), addr(feature), addr(threshold),
        addr(value), addr(node_weight), addr(missing_left), addr(offsets), addr(scales),
        n_trees, n_outputs, addr(phi), addr(path_feature), addr(path_zero),
        addr(path_one), addr(path_weight), path_stride,
    )
    if status:
        raise RuntimeError(f"Mojo TreeSHAP kernel failed validation (status {status})")


def make_synthetic(
    x: np.ndarray,
    background: np.ndarray,
    masks: np.ndarray,
    varying: np.ndarray,
    synthetic: np.ndarray,
) -> None:
    x = _array(x, np.dtype(np.float64), "x")
    background = _array(background, np.dtype(np.float64), "background", ndim=2)
    masks = _array(masks, np.dtype(np.float64), "masks", ndim=2)
    varying = _array(varying, np.dtype(np.int64), "varying")
    synthetic = _array(
        synthetic, np.dtype(np.float64), "synthetic", ndim=2, writable=True
    )
    if len(x) != background.shape[1] or masks.shape[1] != len(varying):
        raise ValueError("coalition input shapes do not agree")
    expected_rows = len(masks) * len(background)
    if synthetic.shape != (expected_rows, background.shape[1]):
        raise ValueError("synthetic output shape does not agree")
    if len(varying) and (
        int(varying.min()) < 0 or int(varying.max()) >= background.shape[1]
    ):
        raise ValueError("varying feature index is out of bounds")
    status = lib().msh_make_synthetic(
        addr(x), addr(background), addr(masks), addr(varying), addr(synthetic),
        len(background), background.shape[1], len(masks), len(varying),
    )
    if status:
        raise RuntimeError(f"Mojo masking kernel failed validation (status {status})")
