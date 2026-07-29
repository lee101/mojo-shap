"""Model-agnostic Kernel SHAP with Mojo coalition materialisation."""

from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np

from ._explanation import Explanation
from ._lib import f64, i64, make_synthetic


def _link_function(link):
    if link == "identity" or link is None:
        return lambda x: np.asarray(x, dtype=np.float64)
    if link == "logit":
        return lambda x: np.log(
            np.clip(x, 1e-15, 1 - 1e-15)
            / np.clip(1 - np.asarray(x), 1e-15, 1)
        )
    if hasattr(link, "f"):
        return lambda x: np.asarray(link.f(x), dtype=np.float64)
    if callable(link):
        return lambda x: np.asarray(link(x), dtype=np.float64)
    raise ValueError("link must be 'identity', 'logit', or a callable link")


def _model_output(value, n_rows: int) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim == 0:
        result = np.full(n_rows, float(result))
    if result.shape[0] != n_rows:
        raise ValueError("model output must have one row per input row")
    return result.reshape(n_rows, -1)


def _exact_masks(m: int) -> np.ndarray:
    rows = []
    for size in range(1, m):
        for chosen in itertools.combinations(range(m), size):
            mask = np.zeros(m, dtype=np.float64)
            mask[list(chosen)] = 1.0
            rows.append(mask)
    return np.asarray(rows, dtype=np.float64)


def _sample_masks(m: int, count: int, rng: np.random.Generator) -> np.ndarray:
    target = min(count, (1 << m) - 2) if m < 63 else count
    masks: dict[bytes, np.ndarray] = {}
    while len(masks) < target:
        if m < 63:
            encoded = int(rng.integers(1, (1 << m) - 1))
            mask = np.asarray(
                [(encoded >> j) & 1 for j in range(m)], dtype=np.float64
            )
        else:
            mask = rng.integers(0, 2, size=m).astype(np.float64)
            if not mask.any() or mask.all():
                continue
        masks[mask.tobytes()] = mask
    return np.asarray(list(masks.values())[:target], dtype=np.float64)


def _kernel_weights(masks: np.ndarray) -> np.ndarray:
    m = masks.shape[1]
    size = masks.sum(axis=1).astype(int)
    return np.asarray(
        [
            (m - 1) / (math.comb(m, int(s)) * s * (m - s))
            for s in size
        ],
        dtype=np.float64,
    )


class KernelExplainer:
    """Kernel SHAP for any vectorised Python prediction function."""

    def __init__(
        self,
        model,
        data,
        feature_names=None,
        link="identity",
        **kwargs,
    ):
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"unexpected keyword arguments: {unknown}")
        self.model = model
        raw_data = np.asarray(data)
        if raw_data.ndim not in (1, 2):
            raise ValueError("data must be a non-empty 1D row or 2D matrix")
        self.data = f64(np.atleast_2d(raw_data))
        if not len(self.data) or not self.data.shape[1]:
            raise ValueError("data must be a non-empty 2D background matrix")
        self.feature_names = feature_names
        self.link = link
        self._link = _link_function(link)
        background_prediction = _model_output(model(self.data), len(self.data))
        self._output_count = background_prediction.shape[1]
        expected = background_prediction.mean(axis=0)
        self.expected_value = (
            float(expected[0]) if self._output_count == 1 else expected
        )
        self._linked_expected = self._link(expected)

    def _coalition_values(
        self,
        x: np.ndarray,
        masks: np.ndarray,
        varying: np.ndarray,
        batch_size: int,
    ) -> np.ndarray:
        result = np.empty((len(masks), self._output_count), dtype=np.float64)
        b = len(self.data)
        d = self.data.shape[1]
        for start in range(0, len(masks), batch_size):
            chunk = f64(masks[start : start + batch_size])
            synthetic = np.empty((len(chunk) * b, d), dtype=np.float64)
            make_synthetic(x, self.data, chunk, varying, synthetic)
            prediction = _model_output(self.model(synthetic), len(synthetic))
            result[start : start + len(chunk)] = prediction.reshape(
                len(chunk), b, self._output_count
            ).mean(axis=1)
        return self._link(result)

    def _explain_row(
        self,
        x: np.ndarray,
        nsamples,
        l1_reg,
        rng: np.random.Generator,
        batch_size: int,
    ) -> np.ndarray:
        varying = i64(np.flatnonzero(np.any(self.data != x, axis=0)))
        m = len(varying)
        phi = np.zeros((self.data.shape[1], self._output_count), dtype=np.float64)
        prediction = self._link(_model_output(self.model(x[None, :]), 1)[0])
        delta = prediction - self._linked_expected
        if m == 0:
            return phi
        if m == 1:
            phi[varying[0]] = delta
            return phi

        if nsamples == "auto":
            requested = 2 * m + 2048
        else:
            if isinstance(nsamples, (bool, np.bool_)):
                raise ValueError("nsamples must be a positive integer or 'auto'")
            requested = int(nsamples)
            if requested != nsamples or requested <= 0:
                raise ValueError("nsamples must be a positive integer or 'auto'")
        all_count = (1 << m) - 2 if m < 63 else math.inf
        masks = (
            _exact_masks(m)
            if all_count <= requested
            else _sample_masks(m, requested, rng)
        )
        weights = _kernel_weights(masks)
        values = self._coalition_values(x, masks, varying, batch_size)

        local = np.empty((m, self._output_count), dtype=np.float64)
        root_weight = np.sqrt(weights)
        for output_index in range(self._output_count):
            selected = np.arange(m)
            if isinstance(l1_reg, str) and l1_reg.startswith("num_features("):
                count = int(l1_reg[len("num_features(") : -1])
                if count < m:
                    from sklearn.linear_model import lars_path

                    adjusted = values[:, output_index] - self._linked_expected[
                        output_index
                    ]
                    size = masks.sum(axis=1)
                    augmented_weight = np.hstack(
                        (weights * (m - size), weights * size)
                    )
                    augmented_root = np.sqrt(augmented_weight)
                    augmented_target = np.hstack(
                        (adjusted, adjusted - delta[output_index])
                    )
                    augmented_masks = np.vstack((masks, masks - 1.0))
                    selected = lars_path(
                        augmented_masks * augmented_root[:, None],
                        augmented_target * augmented_root,
                        max_iter=count,
                    )[1]
            if not len(selected):
                local[:, output_index] = 0.0
                continue
            last = selected[-1]
            design = masks[:, selected[:-1]] - masks[:, last, None]
            target = (
                values[:, output_index]
                - self._linked_expected[output_index]
                - masks[:, last] * delta[output_index]
            )
            coefficients = np.linalg.lstsq(
                design * root_weight[:, None],
                target * root_weight,
                rcond=None,
            )[0]
            local[:, output_index] = 0.0
            local[selected[:-1], output_index] = coefficients
            local[last, output_index] = delta[output_index] - coefficients.sum()
        phi[varying] = local
        return phi

    def shap_values(
        self,
        X,
        nsamples="auto",
        l1_reg="num_features(10)",
        silent=False,
        **kwargs,
    ):
        raw = np.asarray(X)
        if raw.ndim not in (1, 2):
            raise ValueError("X must be a 1D row or 2D matrix")
        was_vector = raw.ndim == 1
        matrix = f64(np.atleast_2d(raw))
        if not len(matrix):
            raise ValueError("X must contain at least one row")
        if matrix.shape[1] != self.data.shape[1]:
            raise ValueError(
                f"X has {matrix.shape[1]} features; background has {self.data.shape[1]}"
            )
        if not (
            l1_reg in (0, 0.0, None, False)
            or (
                isinstance(l1_reg, str)
                and l1_reg.startswith("num_features(")
                and l1_reg.endswith(")")
            )
        ):
            raise NotImplementedError(
                "covered l1_reg modes are 0 and 'num_features(N)'"
            )
        seed = kwargs.pop("seed", 0)
        batch_size = int(kwargs.pop("batch_size", 256))
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if isinstance(l1_reg, str):
            try:
                selected_count = int(l1_reg[len("num_features(") : -1])
            except ValueError as error:
                raise ValueError("num_features(N) requires a positive integer N") from error
            if selected_count <= 0:
                raise ValueError("num_features(N) requires a positive integer N")
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"unexpected keyword arguments: {unknown}")
        rng = np.random.default_rng(seed)
        phi = np.asarray(
            [
                self._explain_row(row, nsamples, l1_reg, rng, batch_size)
                for row in matrix
            ]
        )
        if self._output_count == 1:
            result = phi[:, :, 0]
        else:
            result = phi
        return result[0] if was_vector else result

    def __call__(
        self,
        X,
        nsamples="auto",
        l1_reg="num_features(10)",
        silent=False,
        **kwargs,
    ):
        matrix = f64(np.atleast_2d(X))
        values = self.shap_values(
            matrix, nsamples=nsamples, l1_reg=l1_reg, silent=silent, **kwargs
        )
        expected = np.asarray(self.expected_value)
        if expected.ndim == 0:
            base = np.full(len(matrix), float(expected))
        else:
            base = np.broadcast_to(expected, (len(matrix), len(expected))).copy()
        return Explanation(values, base, matrix, self.feature_names)
