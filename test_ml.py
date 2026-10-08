import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics

CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

@pytest.fixture(scope="module")
def sample_data():
    return pd.DataFrame({
        "age": [39, 28, 45, 23],
        "workclass": ["State-gov", "Private", "Private", "Private"],
        "education": ["Bachelors", "HS-grad", "Bachelors", "HS-grad"],
        "marital-status": ["Never-married", "Married-civ-spouse", "Divorced", "Never-married"],
        "occupation": ["Adm-clerical", "Handlers-cleaners", "Exec-managerial", "Adm-clerical"],
        "relationship": ["Not-in-family", "Husband", "Not-in-family", "Own-child"],
        "race": ["White", "Black", "White", "White"],
        "sex": ["Male", "Male", "Male", "Female"],
        "native-country": ["United-States", "United-States", "United-States", "United-States"],
        "hours-per-week": [40, 40, 60, 20],
        "salary": ["<=50K", ">50K", "<=50K", "<=50K"],
    })


def test_train_model_returns_random_forest(sample_data):
    """train_model should return a fitted RandomForestClassifier."""
    X, y, _, _ = process_data(
        sample_data, categorical_features=CAT_FEATURES, label="salary", training=True
    )
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)


def test_inference_returns_numpy_array(sample_data):
    """inference should return a numpy array with one prediction per row."""
    X, y, _, _ = process_data(
        sample_data, categorical_features=CAT_FEATURES, label="salary", training=True
    )
    model = train_model(X, y)
    preds = inference(model, X)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (len(sample_data),)


def test_compute_model_metrics_perfect_predictions():
    """compute_model_metrics should return 1.0 for all metrics on perfect predictions."""
    y = np.array([0, 1, 0, 1, 1])
    preds = np.array([0, 1, 0, 1, 1])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert precision == 1.0
    assert recall == 1.0
    assert fbeta == 1.0


def test_process_data_output_shape(sample_data):
    """process_data should return X and y with matching row counts."""
    X, y, _, _ = process_data(
        sample_data, categorical_features=CAT_FEATURES, label="salary", training=True
    )
    assert X.shape[0] == len(sample_data)
    assert y.shape[0] == len(sample_data)


def test_process_data_label_is_binary(sample_data):
    """process_data should binarize the label to only 0s and 1s."""
    _, y, _, _ = process_data(
        sample_data, categorical_features=CAT_FEATURES, label="salary", training=True
    )
    assert set(y).issubset({0, 1})
