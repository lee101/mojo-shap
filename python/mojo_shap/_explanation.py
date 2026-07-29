"""The small, array-oriented part of shap.Explanation used by explainers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Explanation:
    values: np.ndarray
    base_values: Any
    data: np.ndarray
    feature_names: Any = None
    output_names: Any = None

    @property
    def shape(self):
        return self.values.shape

    def __len__(self):
        return len(self.values)

    def __getitem__(self, item):
        values = self.values[item]
        data = self.data[item]
        base = np.asarray(self.base_values)
        if base.ndim and base.shape[0] == len(self.values):
            base = base[item]
        return Explanation(
            values, base, data, self.feature_names, self.output_names
        )
