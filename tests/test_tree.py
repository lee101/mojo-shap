import numpy as np
import pytest

shap = pytest.importorskip("shap")
sk_ensemble = pytest.importorskip("sklearn.ensemble")
sk_tree = pytest.importorskip("sklearn.tree")

import mojo_shap


@pytest.fixture(scope="module")
def regression_data():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(300, 6))
    y = (
        2.0 * X[:, 0]
        - X[:, 1] ** 2
        + 0.5 * X[:, 2] * X[:, 3]
        + rng.normal(scale=0.02, size=len(X))
    )
    return X, y


@pytest.mark.parametrize(
    "model",
    [
        sk_tree.DecisionTreeRegressor(max_depth=5, random_state=0),
        sk_ensemble.RandomForestRegressor(
            n_estimators=12, max_depth=6, random_state=0
        ),
        sk_ensemble.ExtraTreesRegressor(
            n_estimators=12, max_depth=6, random_state=0
        ),
        sk_ensemble.GradientBoostingRegressor(
            n_estimators=15, max_depth=3, random_state=0
        ),
    ],
)
def test_regressor_parity(regression_data, model):
    X, y = regression_data
    model.fit(X, y)
    query = X[:9]
    ours = mojo_shap.TreeExplainer(model)
    upstream = shap.TreeExplainer(model)
    actual = ours.shap_values(query)
    expected = np.asarray(upstream.shap_values(query))
    assert actual.shape == expected.shape
    assert np.allclose(actual, expected, atol=2e-9, rtol=2e-9)
    assert ours.expected_value == pytest.approx(
        float(np.asarray(upstream.expected_value).ravel()[0]), abs=2e-9
    )
    assert np.allclose(actual.sum(axis=1) + ours.expected_value, model.predict(query))


@pytest.mark.parametrize(
    "model",
    [
        sk_tree.DecisionTreeClassifier(max_depth=5, random_state=0),
        sk_ensemble.RandomForestClassifier(
            n_estimators=10, max_depth=5, random_state=0
        ),
        sk_ensemble.ExtraTreesClassifier(
            n_estimators=10, max_depth=5, random_state=0
        ),
    ],
)
def test_classifier_parity(model):
    rng = np.random.default_rng(9)
    X = rng.normal(size=(250, 5))
    y = (X[:, 0] + X[:, 1] * X[:, 2] > 0).astype(int)
    model.fit(X, y)
    query = X[:7]
    ours = mojo_shap.TreeExplainer(model)
    upstream = shap.TreeExplainer(model)
    actual = ours.shap_values(query)
    expected = np.asarray(upstream.shap_values(query))
    assert actual.shape == expected.shape
    assert np.allclose(actual, expected, atol=2e-9, rtol=2e-9)
    assert np.allclose(ours.expected_value, upstream.expected_value, atol=2e-9)
    assert np.allclose(
        actual.sum(axis=1) + ours.expected_value, model.predict_proba(query)
    )


def test_single_row_explanation_and_tree_limit(regression_data):
    X, y = regression_data
    model = sk_ensemble.RandomForestRegressor(
        n_estimators=8, max_depth=4, random_state=1
    ).fit(X, y)
    explainer = mojo_shap.TreeExplainer(model, feature_names=list("abcdef"))
    values = explainer.shap_values(X[0])
    assert values.shape == (6,)
    explanation = explainer(X[:3])
    assert explanation.values.shape == (3, 6)
    assert explanation.base_values.shape == (3,)
    limited = explainer.shap_values(X[:2], tree_limit=3, check_additivity=False)
    assert limited.shape == (2, 6)


def test_multioutput_regression_parity():
    rng = np.random.default_rng(10)
    X = rng.normal(size=(180, 4))
    y = np.c_[X[:, 0] + X[:, 1], X[:, 2] * X[:, 3]]
    model = sk_tree.DecisionTreeRegressor(
        max_depth=5, random_state=0
    ).fit(X, y)
    ours = mojo_shap.TreeExplainer(model)
    upstream = shap.TreeExplainer(model)
    actual = ours.shap_values(X[:6])
    expected = np.asarray(upstream.shap_values(X[:6]))
    assert actual.shape == (6, 4, 2)
    assert np.allclose(actual, expected, atol=2e-9, rtol=2e-9)
    assert np.allclose(
        actual.sum(axis=1) + ours.expected_value, model.predict(X[:6])
    )


def test_parallel_threshold_and_simd_tail(regression_data):
    X, y = regression_data
    model = sk_ensemble.RandomForestRegressor(
        n_estimators=8, max_depth=5, random_state=2
    ).fit(X, y)
    ours = mojo_shap.TreeExplainer(model)
    upstream = shap.TreeExplainer(model)

    serial = ours.shap_values(X[:63], check_additivity=False)
    serial_expected = np.asarray(upstream.shap_values(X[:63]))
    assert np.allclose(serial, serial_expected, atol=2e-9, rtol=2e-9)

    parallel = ours.shap_values(X[:64], check_additivity=False)
    parallel_expected = np.asarray(upstream.shap_values(X[:64]))
    assert np.allclose(parallel, parallel_expected, atol=2e-9, rtol=2e-9)

    chunked = ours.shap_values(X[:65], check_additivity=False)
    chunked_expected = np.asarray(upstream.shap_values(X[:65]))
    assert np.allclose(chunked, chunked_expected, atol=2e-9, rtol=2e-9)


def test_unsupported_interventional(regression_data):
    X, y = regression_data
    model = sk_tree.DecisionTreeRegressor().fit(X, y)
    with pytest.raises(NotImplementedError):
        mojo_shap.TreeExplainer(model, data=X[:10])


def test_multiclass_classifier_and_tree_dispatch():
    rng = np.random.default_rng(22)
    X = rng.normal(size=(240, 5))
    y = np.argmax(
        np.c_[X[:, 0] - X[:, 1], X[:, 1] + X[:, 2], -X[:, 0] - X[:, 2]],
        axis=1,
    )
    model = sk_ensemble.RandomForestClassifier(
        n_estimators=9, max_depth=5, random_state=0
    ).fit(X, y)
    ours = mojo_shap.Explainer(model)
    assert isinstance(ours, mojo_shap.TreeExplainer)
    actual = ours.shap_values(X[:8])
    expected = np.asarray(shap.TreeExplainer(model).shap_values(X[:8]))
    assert actual.shape == (8, 5, 3)
    assert np.allclose(actual, expected, atol=2e-9, rtol=2e-9)
    explanation = ours(X[:2])
    assert np.array_equal(explanation.output_names, model.classes_)
    assert np.array_equal(explanation[:1].output_names, model.classes_)


def test_invalid_rank_does_not_cross_ffi(regression_data):
    X, y = regression_data
    model = sk_tree.DecisionTreeRegressor().fit(X, y)
    with pytest.raises(ValueError, match="1D row or 2D matrix"):
        mojo_shap.TreeExplainer(model).shap_values(X[:6].reshape(2, 3, 6))
