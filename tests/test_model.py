"""
Unit tests for the from-scratch LogisticRegression (A3, Task 3 - Objective 3).

Test 1: the model takes the expected input  (a 2-D matrix with the intercept column + the features)
Test 2: the output of the model has the expected shape (one class per row, 4 probabilities per row)

Run from the project root with:   python -m pytest -v
"""
import numpy as np
import pytest

from logistic_regression import LogisticRegression, one_hot

K = 4              # number of price classes
N_FEATURES = 6     # 5 features + 1 intercept column of ones (the same layout we use in the notebook)


def make_small_trained_model(m=80, seed=0):
    """Train on a tiny random dataset so the tests run in a fraction of a second."""
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(m), rng.normal(size=(m, N_FEATURES - 1))])  # intercept + 5 features
    y = rng.integers(0, K, size=m)
    np.random.seed(seed)  # the model starts from random weights, so fix them for a repeatable test
    model = LogisticRegression(k=K, n=N_FEATURES, method="batch", alpha=0.1,
                               max_iter=200, verbose=False)
    model.fit(X, one_hot(y, K))
    return model, X


def test_model_takes_expected_input():
    model, X = make_small_trained_model()

    # the weights must match the expected input layout: (features incl. intercept, classes)
    assert model.W.shape == (N_FEATURES, K)

    # an input with the expected number of columns is accepted
    model.predict(X)

    # an input with a missing column (wrong number of features) must be rejected
    with pytest.raises(ValueError):
        model.predict(X[:, :-1])


def test_model_output_has_expected_shape():
    model, X = make_small_trained_model()
    m = X.shape[0]

    # predict -> one class label per row, and only the labels 0..3 are possible
    pred = model.predict(X)
    assert pred.shape == (m,)
    assert set(np.unique(pred)).issubset(set(range(K)))

    # predict_proba -> 4 probabilities per row that add up to 1
    proba = model.predict_proba(X)
    assert proba.shape == (m, K)
    assert np.allclose(proba.sum(axis=1), 1.0)

    