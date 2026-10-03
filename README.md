# Loan Eligibility Prediction

A machine learning web application that predicts loan eligibility based on applicant information. The application uses a Scikit-learn pipeline with data preprocessing and an SVM classifier, and provides a Flask web interface and REST API.

## 🚀 Live Demo

**Render:** https://loan-eligibility-prediction-uwj0.onrender.com

## 📌 GitHub Repository

**GitHub:** https://github.com/nantha2804/loan-eligibility-prediction

## ✨ Features

* Loan eligibility prediction using Machine Learning
* Flask web application
* REST API for predictions
* Data preprocessing using Scikit-learn
* Missing-value handling
* Categorical feature encoding
* Numerical feature scaling
* SVM classification
* Input validation
* Health-check endpoint
* Automated tests with Pytest
* UV-based dependency management
* Deployment on Render

## 🛠️ Technologies Used

* Python
* Pandas
* Scikit-learn
* Flask
* Gunicorn
* Pytest
* uv
* Git & GitHub
* Render

## 🤖 Machine Learning

The application uses an SVM classifier to predict the `Loan_Status` target.

### Preprocessing

* Numerical missing values → Median imputation
* Categorical missing values → Most-frequent imputation
* Categorical features → One-hot encoding
* Numerical features → Standard scaling
* Class imbalance → Balanced class weights

The model uses a stratified 80/20 train-test split for evaluation and is then retrained on the complete dataset for deployment.

## 📊 Input Features

The prediction model uses:

* Gender
* Married
* Dependents
* Education
* Self_Employed
* ApplicantIncome
* CoapplicantIncome
* LoanAmount
* Loan_Amount_Term
* Credit_History
* Property_Area

`Loan_ID` is excluded because it is only an identifier.

## 🌐 API Endpoints

### Health Check

```text
GET /health
```

Example response:

```json
{
  "status": "ok"
}
```

### Loan Prediction

```text
POST /predict
```

Example request:

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

Example response:

```json
{
  "prediction": "Y",
  "eligibility": "Eligible"
}
```

## 💻 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/nantha2804/loan-eligibility-prediction.git
cd loan-eligibility-prediction
```

### 2. Install dependencies

Make sure `uv` is installed, then run:

```bash
uv sync
```

### 3. Start the Flask application

```bash
uv run flask --app app run --debug
```

Open:

```text
http://127.0.0.1:5000
```

### 4. Run tests

```bash
uv run pytest
```

## 📁 Project Structure

```text
loan-eligibility-prediction/
│
├── app.py
├── model.py
├── LoanData.csv
├── cleaned.xlsx
├── templates/
│   └── index.html
├── tests/
│   └── test_app.py
├── Loan Eligible Status Project.ipynb
├── pyproject.toml
├── uv.lock
├── render.yaml
├── Dockerfile
├── .gitignore
└── README.md
```

## ⚠️ Disclaimer

This project is an educational machine-learning demonstration. It is not financial advice, a credit assessment, or an automated lending decision.

## 👨‍💻 Author

**Nantha Kumar**

Computer Science Graduate
Generative AI & Agentic AI | Python | Machine Learning
