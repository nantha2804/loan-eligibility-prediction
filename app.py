from __future__ import annotations

from functools import lru_cache
from math import isfinite
from pathlib import Path
from typing import Any

import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.pipeline import Pipeline

from model import FEATURES, train_model


DATASET_PATH = Path(__file__).resolve().parent / "LoanData.csv"


CHOICES = {
    "Gender": {"Male", "Female"},
    "Married": {"Yes", "No"},
    "Dependents": {"0", "1", "2", "3+"},
    "Education": {"Graduate", "Not Graduate"},
    "Self_Employed": {"Yes", "No"},
    "Property_Area": {"Urban", "Semiurban", "Rural"},
}


NON_NEGATIVE_FIELDS = {
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
}


@lru_cache(maxsize=1)
def load_model() -> Pipeline:
    """Train and cache the ML model."""

    trained_model, _ = train_model(DATASET_PATH)

    return trained_model


def validate_applicant(
    payload: Any,
) -> tuple[dict[str, Any] | None, str | None]:
    """Validate applicant input."""

    if not isinstance(payload, dict):
        return None, "Request body must be a JSON object."

    unexpected = sorted(
        set(payload).difference(FEATURES)
    )

    if unexpected:
        return None, (
            f"Unexpected fields: {', '.join(unexpected)}."
        )

    missing = [
        feature
        for feature in FEATURES
        if feature not in payload
    ]

    if missing:
        return None, (
            "Missing required fields: "
            + ", ".join(missing)
            + "."
        )

    applicant: dict[str, Any] = {}

    for feature in FEATURES:
        value = payload[feature]

        # Categorical fields
        if feature in CHOICES:
            normalized = str(value)

            if normalized not in CHOICES[feature]:
                return None, (
                    f"Invalid value for {feature}."
                )

            applicant[feature] = normalized
            continue

        # Numeric fields
        if isinstance(value, bool):
            return None, f"{feature} must be a number."

        try:
            number = float(value)
        except (TypeError, ValueError):
            return None, f"{feature} must be a number."

        if not isfinite(number):
            return None, (
                f"{feature} must be a finite number."
            )

        if feature in NON_NEGATIVE_FIELDS and number < 0:
            return None, (
                f"{feature} cannot be negative."
            )

        if feature == "Loan_Amount_Term" and number <= 0:
            return None, (
                "Loan_Amount_Term must be greater than zero."
            )

        if feature == "Credit_History" and number not in {0, 1}:
            return None, (
                "Credit_History must be 0 or 1."
            )

        applicant[feature] = number

    return applicant, None


def create_app() -> Flask:
    """Create and configure the Flask application."""

    flask_app = Flask(__name__)

    @flask_app.get("/")
    def index():
        return render_template(
            "index.html",
            features=FEATURES,
        )

    @flask_app.get("/health")
    def health():
        return jsonify(
            {
                "status": "ok"
            }
        )

    @flask_app.post("/predict")
    def predict():
        payload = request.get_json(silent=True)

        applicant, error = validate_applicant(payload)

        if error:
            return jsonify(
                {
                    "error": error
                }
            ), 400

        prediction = load_model().predict(
            pd.DataFrame([applicant])
        )[0]

        eligible = str(prediction) == "Y"

        return jsonify(
            {
                "prediction": str(prediction),
                "eligibility": (
                    "Eligible"
                    if eligible
                    else "Not eligible"
                ),
            }
        )

    return flask_app


app = create_app()