from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


TARGET = "Loan_Status"

FEATURES = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Property_Area",
]


def build_pipeline(data: pd.DataFrame) -> Pipeline:
    """Build the preprocessing and machine-learning pipeline."""

    categorical_features = data.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    numeric_features = data.select_dtypes(
        exclude=["object", "string"]
    ).columns.tolist()

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
        ]
    )

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                SVC(
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def train_model(dataset_path: Path) -> tuple[Pipeline, dict[str, float]]:
    """Train the loan eligibility model and return metrics."""

    frame = pd.read_csv(dataset_path)

    required_columns = set(FEATURES + [TARGET])

    missing_columns = required_columns.difference(frame.columns)

    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: "
            + ", ".join(sorted(missing_columns))
            + "."
        )

    features = frame[FEATURES].copy()
    target = frame[TARGET].astype(str)

    if target.nunique() < 2:
        raise ValueError(
            "Training data must contain at least two loan status classes."
        )

    train_features, test_features, train_target, test_target = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    model = build_pipeline(train_features)

    # Train on training data
    model.fit(train_features, train_target)

    # Evaluate on unseen test data
    holdout_predictions = model.predict(test_features)

    metrics = {
        "accuracy": float(
            accuracy_score(test_target, holdout_predictions)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                test_target,
                holdout_predictions,
            )
        ),
    }

    # Retrain on the complete dataset
    model.fit(features, target)

    return model, metrics