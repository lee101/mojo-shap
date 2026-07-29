import numpy as np
import pytest

shap = pytest.importorskip("shap")

import mojo_shap


def test_exact_linear_parity():
    rng = np.random.default_rng(12)
    background = rng.normal(size=(18, 5))
    weights = np.array([1.5, -2.0, 0.0, 0.75, 3.0])

    def model(X):
        return X @ weights + 0.4

    query = rng.normal(size=(3, 5))
    ours = mojo_shap.KernelExplainer(model, background)
    upstream = shap.KernelExplainer(model, background)
    actual = ours.shap_values(query, nsamples=30, l1_reg=0)
    expected = np.asarray(
        upstream.shap_values(query, nsamples=30, l1_reg=0, silent=True)
    )
    assert np.allclose(actual, expected, atol=2e-9, rtol=2e-9)
    assert ours.expected_value == pytest.approx(upstream.expected_value)
    assert np.allclose(actual.sum(axis=1) + ours.expected_value, model(query))


def test_exact_nonlinear_parity():
    rng = np.random.default_rng(13)
    background = rng.normal(size=(12, 4))

    def model(X):
        return X[:, 0] * X[:, 1] + np.sin(X[:, 2]) + X[:, 3] ** 2

    query = rng.normal(size=(2, 4))
    actual = mojo_shap.KernelExplainer(model, background).shap_values(
        query, nsamples=14, l1_reg=0
    )
    expected = np.asarray(
        shap.KernelExplainer(model, background).shap_values(
            query, nsamples=14, l1_reg=0, silent=True
        )
    )
    assert np.allclose(actual, expected, atol=2e-8, rtol=2e-8)


def test_multioutput_and_logit_additivity():
    rng = np.random.default_rng(14)
    background = rng.normal(size=(10, 3))

    def model(X):
        z = X @ np.array([0.8, -0.5, 1.2])
        p = 1 / (1 + np.exp(-z))
        return np.c_[1 - p, p]

    query = rng.normal(size=(2, 3))
    explainer = mojo_shap.KernelExplainer(model, background, link="logit")
    values = explainer.shap_values(query, nsamples=6, l1_reg=0)
    assert values.shape == (2, 3, 2)
    linked_prediction = np.log(model(query) / (1 - model(query)))
    linked_expected = np.log(
        np.asarray(explainer.expected_value)
        / (1 - np.asarray(explainer.expected_value))
    )
    assert np.allclose(values.sum(axis=1) + linked_expected, linked_prediction)
    explanation = explainer(query, nsamples=6, l1_reg=0)
    assert explanation.base_values.shape == (2, 2)


def test_constant_feature_gets_zero():
    background = np.array([[0.0, 2.0, 1.0], [1.0, 2.0, 3.0]])

    def model(X):
        return X[:, 0] + X[:, 1] * X[:, 2]

    values = mojo_shap.KernelExplainer(model, background).shap_values(
        np.array([[4.0, 2.0, 5.0]]), nsamples=6, l1_reg=0
    )
    assert values[0, 1] == 0.0


def test_sampled_mode_is_additive():
    rng = np.random.default_rng(15)
    background = rng.normal(size=(8, 16))
    weights = rng.normal(size=16)

    def model(X):
        return np.tanh(X @ weights)

    query = rng.normal(size=(2, 16))
    explainer = mojo_shap.KernelExplainer(model, background)
    values = explainer.shap_values(
        query, nsamples=80, l1_reg=0, seed=3, batch_size=20
    )
    assert np.all(np.isfinite(values))
    assert np.allclose(
        values.sum(axis=1) + explainer.expected_value,
        model(query),
        atol=1e-10,
    )


def test_default_feature_selection_and_vector_input():
    rng = np.random.default_rng(16)
    background = rng.normal(size=(6, 12))
    coefficient = np.r_[np.arange(1.0, 5.0), np.zeros(8)]

    def model(X):
        return X @ coefficient

    explainer = mojo_shap.KernelExplainer(model, background)
    values = explainer.shap_values(rng.normal(size=12), nsamples=80, seed=2)
    assert values.shape == (12,)
    assert np.count_nonzero(values) <= 10


def test_explainer_dispatch():
    background = np.zeros((2, 2))
    explainer = mojo_shap.Explainer(lambda X: X.sum(axis=1), background)
    assert isinstance(explainer, mojo_shap.KernelExplainer)


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        ("X", np.zeros((1, 1, 2)), "1D row or 2D matrix"),
        ("nsamples", 0, "positive integer"),
        ("batch_size", 0, "positive"),
    ],
)
def test_invalid_dimensions_do_not_cross_ffi(argument, value, message):
    explainer = mojo_shap.KernelExplainer(
        lambda X: X.sum(axis=1), np.zeros((2, 2))
    )
    kwargs = {"X": np.ones((1, 2)), "nsamples": 2, "l1_reg": 0}
    kwargs[argument] = value
    with pytest.raises(ValueError, match=message):
        explainer.shap_values(**kwargs)


def test_nonrepresentable_integer_input_is_rejected():
    background = np.array([[0, 0], [1, 1]], dtype=np.int64)
    explainer = mojo_shap.KernelExplainer(
        lambda X: X.sum(axis=1), background
    )
    query = np.array([2**53 + 1, 0], dtype=np.int64)
    with pytest.raises(ValueError, match="silent narrowing"):
        explainer.shap_values(query)


def test_explanation_slicing_preserves_metadata():
    explanation = mojo_shap.KernelExplainer(
        lambda X: X.sum(axis=1),
        np.zeros((2, 2)),
        feature_names=["a", "b"],
    )(np.ones((3, 2)), nsamples=2, l1_reg=0)
    sliced = explanation[1:]
    assert sliced.values.shape == (2, 2)
    assert sliced.base_values.shape == (2,)
    assert sliced.data.shape == (2, 2)
    assert sliced.feature_names == ["a", "b"]
