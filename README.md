# mojo-shap

`mojo-shap` is a focused port of SHAP's two core explanation algorithms to
[Mojo](https://www.modular.com/mojo). It provides exact path-dependent
TreeSHAP for common scikit-learn trees and model-agnostic Kernel SHAP behind a
Python API matching the covered parts of `shap`.

```python
import numpy as np
import mojo_shap as shap
from sklearn.ensemble import RandomForestRegressor

X_train = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y_train = np.array([0.0, 1.0, 1.0, 3.0])
X_test = np.array([[0.25, 0.75]])

model = RandomForestRegressor(n_estimators=100, random_state=0).fit(X_train, y_train)
explainer = shap.TreeExplainer(model)
explanation = explainer(X_test)

print(explanation.values)
print(explanation.base_values)
```

The same import change works for Kernel SHAP:

```python
explainer = shap.KernelExplainer(model.predict, X_background)
values = explainer.shap_values(X_test, nsamples=2048)
```

## Coverage

| API | Covered |
| --- | --- |
| `TreeExplainer` | exact `tree_path_dependent` SHAP values for fitted scikit-learn `DecisionTreeRegressor`, `DecisionTreeClassifier`, `RandomForestRegressor`, `RandomForestClassifier`, `ExtraTreesRegressor`, `ExtraTreesClassifier`, and `GradientBoostingRegressor` |
| `KernelExplainer` | scalar or multi-output vectorized callables; identity and logit links; exact coalition enumeration; sampled coalitions; `l1_reg=0` and `num_features(N)` selection |
| `Explainer` | dispatch to the covered tree or kernel explainer |
| `Explanation` | `values`, `base_values`, `data`, `feature_names`, `output_names`, slicing |

Tree regression supports single- and multi-output targets. Tree classification
supports a single target with any number of classes. Kernel SHAP accepts dense
NumPy-compatible background and query matrices.

Not covered: interventional TreeSHAP, interaction values, probability/log-loss
TreeSHAP transforms, categorical or sparse maskers, partition/deep/linear
explainers, plots, text/image explainers, XGBoost/LightGBM/CatBoost model
parsers, GradientBoostingClassifier, or upstream's legacy utility modules.
Unsupported modes raise an explicit error.

## Install

Linux x86-64 and Pixi are required. The repository pins the Mojo nightly and
all test dependencies:

```bash
git clone https://github.com/lee101/mojo-shap.git
cd mojo-shap
pixi install
pixi run build
pixi run test
```

Run Python examples inside the environment:

```bash
pixi run python your_script.py
```

`pixi run build` writes `dist/libmojo-shap.so`. The Python package can also use
a compatible library selected with `MOJO_SHAP_LIB`; standalone wheel
distribution is not currently provided.

## Correctness

`pixi run test` runs 25 tests, including attribution comparisons against the
real upstream `shap` package. The
suite checks attribution-by-attribution parity for four regressor families,
three classifier families (including multiclass output), exact linear and
nonlinear Kernel SHAP, multi-output and logit behavior, feature selection,
input validation, explanation slicing, SIMD remainder handling, the TreeSHAP
parallel threshold, and both algorithms' local-accuracy identity:

```text
sum(shap_values) + expected_value == model_output
```

The comparison tolerance for ordinary TreeSHAP and exact Kernel SHAP is
generally `2e-9`.

## Performance

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz.
Times are the best of four warm runs. The forest benchmark fixes upstream
scikit-learn to one prediction thread. `mojo-shap` parallelizes independent
rows only when the TreeSHAP workload crosses its size threshold.

| case | mojo-shap | shap | result |
| --- | ---: | ---: | ---: |
| TreeSHAP, random forest with 100 depth-10 trees, 1,000 rows | 451.1 ms | 13,932.5 ms | 30.89x faster |
| exact Kernel SHAP, 10 rows, 10 features, 100 background rows | 160.4 ms | 788.4 ms | 4.92x faster |

TreeSHAP vectorizes path-buffer copies, output clearing, and multi-output leaf
accumulation. Large row batches are split into at most 64 parallel work chunks;
small workloads stay serial to avoid thread-launch overhead. Kernel SHAP wins
because coalition materialization is one compiled Mojo call per batch rather
than repeated NumPy indexing and copying. The benchmark includes model
evaluation and weighted regression; it does not time an isolated toy kernel.

There is no GPU implementation. Both kernels run on the CPU.

## How it works

All compiled code lives in one `src/kernels.mojo` compilation unit. Python
flattens every tree into contiguous arrays for child indices, split features,
thresholds, leaf values, node cover, and per-tree scaling. Mojo walks the
decision paths and applies the polynomial-time path extend/unwind algorithm,
accumulating a row-major `(samples, features, outputs)` result.

Kernel SHAP identifies features that differ from the background, constructs
the coalition masks, and sends each mask batch to Mojo. Mojo expands it into a
contiguous row-major synthetic matrix `(coalitions * background, features)`.
The user's Python model evaluates that matrix once per batch; Python then
averages background predictions and solves the constrained kernel-weighted
least-squares system.

The C boundary is deliberately small:

```text
NumPy-owned contiguous float64/int64 buffers
        |
        | ctypes: address and dimensions only
        v
dist/libmojo-shap.so
```

Buffers cross the C ABI as 64-bit integer addresses because exported Mojo
functions cannot have parametric pointer origins. Before each synchronous call,
the Python wrapper checks rank, dtype, C contiguity, writability, lengths, and
index bounds. The Mojo entry point rejects null pointers and invalid dimensions,
then rebuilds the addresses as
`UnsafePointer[..., AnyOrigin[mut=True]]`. The arrays remain strongly referenced
for the call; Mojo does not retain or free them.

## License

MIT
