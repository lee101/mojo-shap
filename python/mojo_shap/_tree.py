"""Exact path-dependent TreeSHAP for scikit-learn tree ensembles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ._explanation import Explanation
from ._lib import f64, i64, tree_shap

_PARALLEL_MIN_ROWS = 8
_PARALLEL_MIN_ROW_TREES = 512
_PARALLEL_MAX_TASKS = 64


@dataclass
class _PackedTrees:
    children_left: np.ndarray
    children_right: np.ndarray
    feature: np.ndarray
    threshold: np.ndarray
    value: np.ndarray
    node_weight: np.ndarray
    missing_left: np.ndarray
    offsets: np.ndarray
    scales: np.ndarray
    base_offset: np.ndarray
    n_outputs: int
    n_features: int
    max_depth: int


def _normalised_classifier_values(tree) -> np.ndarray:
    raw = np.asarray(tree.value, dtype=np.float64)
    if raw.shape[1] != 1:
        raise NotImplementedError("multi-output classification is not covered")
    values = raw[:, 0, :]
    totals = values.sum(axis=1, keepdims=True)
    return np.divide(values, totals, out=np.zeros_like(values), where=totals != 0)


def _regression_values(tree) -> np.ndarray:
    return np.asarray(tree.value, dtype=np.float64)[:, :, 0]


def _tree_parts(tree, values: np.ndarray):
    missing = getattr(tree, "missing_go_to_left", None)
    if missing is None:
        missing = np.zeros(tree.node_count, dtype=np.int64)
    return (
        i64(tree.children_left),
        i64(tree.children_right),
        i64(tree.feature),
        f64(tree.threshold),
        f64(values),
        f64(tree.weighted_n_node_samples),
        i64(missing),
    )


def _pack_model(model) -> _PackedTrees:
    name = type(model).__name__
    n_features = int(getattr(model, "n_features_in_", 0))
    if not n_features:
        raise TypeError("model must be a fitted scikit-learn tree estimator")

    entries: list[tuple[Any, np.ndarray, float]] = []
    base_offset: np.ndarray

    if hasattr(model, "tree_"):
        tree = model.tree_
        if hasattr(model, "classes_"):
            values = _normalised_classifier_values(tree)
        else:
            values = _regression_values(tree)
        entries.append((tree, values, 1.0))
        base_offset = np.zeros(values.shape[1], dtype=np.float64)
    elif name in {
        "RandomForestRegressor",
        "ExtraTreesRegressor",
        "RandomForestClassifier",
        "ExtraTreesClassifier",
    }:
        estimators = list(model.estimators_)
        scale = 1.0 / len(estimators)
        classifier = hasattr(model, "classes_")
        for estimator in estimators:
            values = (
                _normalised_classifier_values(estimator.tree_)
                if classifier
                else _regression_values(estimator.tree_)
            )
            entries.append((estimator.tree_, values, scale))
        base_offset = np.zeros(entries[0][1].shape[1], dtype=np.float64)
    elif name == "GradientBoostingRegressor":
        for estimator in np.asarray(model.estimators_).ravel():
            entries.append(
                (estimator.tree_, _regression_values(estimator.tree_), model.learning_rate)
            )
        probe = np.zeros((1, n_features), dtype=np.float64)
        base_offset = np.asarray(model._raw_predict_init(probe)[0], dtype=np.float64)
    else:
        raise TypeError(
            "covered models are DecisionTree, RandomForest, ExtraTrees, and "
            "GradientBoostingRegressor"
        )

    n_outputs = entries[0][1].shape[1]
    if any(values.shape[1] != n_outputs for _, values, _ in entries):
        raise ValueError("all trees must have the same number of outputs")

    parts = [_tree_parts(tree, values) for tree, values, _ in entries]
    sizes = np.array([len(part[0]) for part in parts], dtype=np.int64)
    offsets = np.r_[0, np.cumsum(sizes[:-1])].astype(np.int64)
    return _PackedTrees(
        *(np.ascontiguousarray(np.concatenate([p[i] for p in parts])) for i in range(7)),
        offsets=np.ascontiguousarray(offsets),
        scales=f64([entry[2] for entry in entries]),
        base_offset=f64(base_offset),
        n_outputs=n_outputs,
        n_features=n_features,
        max_depth=max(int(tree.max_depth) for tree, _, _ in entries),
    )


class TreeExplainer:
    """Drop-in TreeExplainer for the covered scikit-learn estimators.

    Only ``feature_perturbation="tree_path_dependent"`` and raw model output
    are supported. This is the exact polynomial-time TreeSHAP algorithm.
    """

    def __init__(
        self,
        model,
        data=None,
        model_output="raw",
        feature_perturbation="auto",
        feature_names=None,
        approximate=False,
        link=None,
        linearize_link=None,
    ):
        if data is not None:
            raise NotImplementedError(
                "background-data/interventional TreeSHAP is not covered"
            )
        if model_output != "raw":
            raise NotImplementedError("only model_output='raw' is covered")
        if feature_perturbation not in ("auto", "tree_path_dependent"):
            raise NotImplementedError(
                "only feature_perturbation='tree_path_dependent' is covered"
            )
        if approximate:
            raise NotImplementedError("approximate Saabas values are not covered")
        self.model = model
        self.feature_names = feature_names
        self._trees = _pack_model(model)
        self.expected_value = self._base_value(len(self._trees.offsets))

    def _base_value(self, tree_limit: int):
        packed = self._trees
        roots = packed.value[packed.offsets[:tree_limit]]
        value = packed.base_offset + np.sum(
            roots * packed.scales[:tree_limit, None], axis=0
        )
        return float(value[0]) if packed.n_outputs == 1 else value

    def shap_values(
        self,
        X,
        y=None,
        tree_limit=None,
        approximate=False,
        check_additivity=True,
        from_call=False,
    ):
        if approximate:
            raise NotImplementedError("approximate Saabas values are not covered")
        was_vector = np.asarray(X).ndim == 1
        raw = np.asarray(X)
        if raw.ndim not in (1, 2):
            raise ValueError("X must be a 1D row or 2D matrix")
        matrix = f64(np.atleast_2d(raw))
        packed = self._trees
        if not len(matrix):
            raise ValueError("X must contain at least one row")
        if matrix.shape[1] != packed.n_features:
            raise ValueError(
                f"X has {matrix.shape[1]} features; model expects {packed.n_features}"
            )
        limit = len(packed.offsets) if tree_limit in (None, -1) else int(tree_limit)
        if not 0 < limit <= len(packed.offsets):
            raise ValueError("tree_limit is outside the fitted ensemble")

        phi = np.empty(
            (len(matrix), packed.n_features, packed.n_outputs), dtype=np.float64
        )
        path_size = (packed.max_depth + 2) * (packed.max_depth + 3) // 2 + 1
        parallel = (
            len(matrix) >= _PARALLEL_MIN_ROWS
            and len(matrix) * limit >= _PARALLEL_MIN_ROW_TREES
        )
        scratch_rows = min(len(matrix), _PARALLEL_MAX_TASKS) if parallel else 1
        path_feature = np.empty(scratch_rows * path_size, dtype=np.int64)
        path_buffers = np.empty((3, scratch_rows * path_size), dtype=np.float64)
        path_zero, path_one, path_weight = path_buffers
        tree_shap(
            matrix,
            packed.children_left,
            packed.children_right,
            packed.feature,
            packed.threshold,
            packed.value,
            packed.node_weight,
            packed.missing_left,
            packed.offsets,
            packed.scales,
            limit,
            packed.n_outputs,
            phi,
            path_feature,
            path_zero,
            path_one,
            path_weight,
            path_size if parallel else 0,
        )

        if packed.n_outputs == 1:
            result = phi[:, :, 0]
        else:
            result = phi
        if check_additivity and limit == len(packed.offsets):
            prediction = (
                self.model.predict_proba(matrix)
                if hasattr(self.model, "predict_proba")
                else self.model.predict(matrix)
            )
            summed = result.sum(axis=1) + self._base_value(limit)
            if not np.allclose(summed, prediction, rtol=1e-5, atol=1e-7):
                error = float(np.max(np.abs(summed - prediction)))
                raise RuntimeError(f"TreeSHAP additivity check failed (max error {error:g})")
        return result[0] if was_vector else result

    def __call__(
        self,
        X,
        y=None,
        interactions=False,
        check_additivity=True,
        approximate=False,
    ):
        if interactions:
            raise NotImplementedError("SHAP interaction values are not covered")
        values = self.shap_values(
            X,
            y=y,
            approximate=approximate,
            check_additivity=check_additivity,
            from_call=True,
        )
        matrix = np.asarray(X)
        n_rows = 1 if matrix.ndim == 1 else len(matrix)
        base = np.asarray(self.expected_value)
        if base.ndim == 0:
            base_values = np.full(n_rows, float(base))
        else:
            base_values = np.broadcast_to(base, (n_rows, len(base))).copy()
        return Explanation(
            np.asarray(values),
            base_values,
            matrix,
            self.feature_names,
            getattr(self.model, "classes_", None),
        )
