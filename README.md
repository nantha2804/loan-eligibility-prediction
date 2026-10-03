# Loan Eligibility Prediction

A Flask web app and JSON API that predicts loan eligibility from the included applicant dataset. The scikit-learn pipeline imputes missing values, one-hot encodes categorical fields, scales numeric fields, and trains an SVM classifier. `Loan_ID` is excluded because it is only an identifier.

> This is an educational machine-learning demo, not financial advice, a credit assessment, or an automated lending decision.

## Stack

Python, uv, Pandas, NumPy (through Pandas/scikit-learn), scikit-learn, Flask, Gunicorn, Docker, GitHub Actions, and Render.

## Run locally

Install [uv](https://docs.astral.sh/uv/) and Python 3.10 or later, then run:

```bash
uv sync
uv run flask --app app run --debug
```

Open <http://127.0.0.1:5000>. The model is trained from `LoanData.csv` on the first prediction request. To run the test suite:

```bash
uv run pytest
```

To install the original notebook's analysis dependencies as well:

```bash
uv sync --extra notebook
uv run jupyter notebook
```

The prediction API accepts JSON at `POST /predict`:

```json
{
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
  "Property_Area": "Urban"
}
```

The response contains the model class (`Y` or `N`) and a readable eligibility label. `GET /health` is available for health checks. Invalid or incomplete input receives a `400` JSON response.
`requirements.txt` is a pip-compatible export of the uv lockfile and includes the notebook extra.
Regenerate it after dependency changes with `uv export --locked --all-extras --no-hashes --format requirements-txt --output-file requirements.txt`.

## Docker

Build and run the container:

```bash
docker build -t loan-eligibility-prediction .
docker run --rm -p 8000:8000 loan-eligibility-prediction
```

Open <http://localhost:8000>. The container uses the committed `uv.lock` for reproducible dependency installation and runs Gunicorn as a non-root user.

## Deploy to Render

1. Push this repository to GitHub.
2. In Render, choose **New + → Blueprint** and connect the repository.
3. Render reads `render.yaml`, syncs the locked dependencies with uv, starts Gunicorn, and checks `/health`.

## GitHub Actions

The workflow in `.github/workflows/ci.yml` runs the tests on pushes and pull requests. Dependencies are defined in `pyproject.toml` and resolved in `uv.lock`.

## Dataset and evaluation

`LoanData.csv` contains applicant attributes and the historical `Loan_Status` target (`Y`/`N`). The training pipeline uses a stratified 80/20 holdout split to calculate accuracy and balanced accuracy, then retrains the deployment model on the full dataset. The notebook preserves the exploratory analysis and experiments.

## Author

Nantha Kumar
