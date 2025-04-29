# 📱 Telecom Customer Churn Prediction

![Tele Comm Png](https://github.com/user-attachments/assets/3593a5ae-87fe-4b4d-894f-b861e37dc85c)

---

## 📂 Project Overview

This project predicts whether a telecom customer is likely to **churn** (leave the service) based on their activity and demographic data.

It follows a modular, production-grade **MLOps architecture** and includes:
- Full pipeline orchestration
- Experiment tracking with metrics
- Model versioning
- Containerization & CI/CD deployment

---

## 🏧 Architecture Summary

```
Data Ingestion → Data Validation → Data Transformation → Model Training → Evaluation → Prediction API
                            └──> Dockerized & Deployed via GitHub Actions + AWS
```

---

## 🚀 Tech Stack

| Component        | Tools Used                             |
|------------------|-----------------------------------------|
| Programming Lang | Python 3.10                             |
| ML Models        | Scikit-learn (Logistic, Tree-based)     |
| Packaging        | Joblib, Pickle                          |
| Experimentation  | Manual metric tracking (JSON)          |
| Serving          | FastAPI                                 |
| UI               | Jinja2 + HTML Templates                 |
| DevOps           | Docker, GitHub Actions, AWS ECR         |

---

## 📂 Folder Structure

```
.
├── .github/workflows/         # CI/CD YAML
├── config/                    # YAML configuration files
├── research/                  # Jupyter notebooks (EDA, experiments)
├── src/mlproject/            # Modular pipeline code
│   ├── components/            # Pipeline building blocks
│   ├── pipeline/              # Stages (ingest, train, eval)
│   ├── config/                # Configuration loader
│   ├── entities/              # Config dataclasses
│   ├── utils/                 # Utility functions
├── artifacts/                # Outputs: models, metrics, transformed data
├── schema.yaml               # Feature schema for validation
├── params.yaml               # Model parameters
├── Dockerfile                # Container build
├── app.py                    # FastAPI inference server
├── templates/                # HTML UI
├── README.md                 # Project documentation
```

---

## 💡 Key Features

- ✅ Modular pipeline (ingest, validate, transform, train, evaluate)
- ✅ Clean training/testing separation
- ✅ Metric logging to `metrics.json`
- ✅ Reusable label encoders + preprocessors
- ✅ Real-time prediction with FastAPI UI
- ✅ Fully containerized with Docker
- ✅ Automated CI/CD to AWS ECR using GitHub Actions

---

## 💠 Setup Instructions

### Clone & Install
```bash
git clone https://github.com/your-username/telecom-churn-prediction.git
cd telecom-churn-prediction
pip install -r requirements.txt
```

### Run Main Pipelines
```bash
python main.py --stage data_ingestion
python main.py --stage data_validation
python main.py --stage data_transformation
python main.py --stage model_training
python main.py --stage model_evaluation
```

### Launch API
```bash
uvicorn app:app --reload
```
Visit [http://localhost:8000](http://localhost:8000)

---

## 🐳 Docker Support

```bash
docker build -t telecom-churn-app .
docker run -p 8000:8000 telecom-churn-app
```

---

## ⚙️ GitHub Actions CI/CD

Your pipeline:
- Runs on every push to `main`
- Lints and tests code
- Builds and pushes Docker image to AWS ECR
- Pulls and runs on self-hosted EC2 Docker instance

CI/CD file: `.github/workflows/cicd.yaml`

---

## 🧠 Model Insights

- Evaluation metrics are saved in `artifacts/model_evaluation/metrics.json`
- Supports **binary classification** with class labels: `Churn` or `No Churn`

---

## 📅 Future Improvements

- Integrate MLflow or Weights & Biases for experiment tracking
- Add Evidently AI for monitoring production drift
- Add Prometheus & Grafana dashboards
- Automate with DVC pipelines
- Expand UI to show confidence scores

---

## 📄 License

Distributed under the MIT License.

---
