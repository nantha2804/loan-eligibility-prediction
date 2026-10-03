from app import app


SAMPLE_APPLICANT = {
    "Gender": "Male",
    "Married": "Yes",
    "Dependents": "0",
    "Education": "Graduate",
    "Self_Employed": "No",
    "ApplicantIncome": 5000,
    "CoapplicantIncome": 0,
    "LoanAmount": 150,
    "Loan_Amount_Term": 360,
    "Credit_History": 1,
    "Property_Area": "Urban",
}


def test_health_check():
    response = app.test_client().get("/health")

    assert response.status_code == 200
    assert response.json == {"status": "ok"}


def test_homepage_renders_prediction_form():
    response = app.test_client().get("/")

    assert response.status_code == 200
    assert b"Loan Eligibility Predictor" in response.data


def test_predict_rejects_missing_fields():
    response = app.test_client().post("/predict", json={})

    assert response.status_code == 400
    assert "Missing required fields" in response.json["error"]


def test_predict_uses_trained_pipeline():
    response = app.test_client().post("/predict", json=SAMPLE_APPLICANT)

    assert response.status_code == 200
    assert response.json["prediction"] in {"Y", "N"}
    assert response.json["eligibility"] in {"Eligible", "Not eligible"}


def test_predict_rejects_invalid_values():
    applicant = {**SAMPLE_APPLICANT, "Credit_History": 2}

    response = app.test_client().post("/predict", json=applicant)

    assert response.status_code == 400
    assert "Credit_History must be 0 or 1" in response.json["error"]
