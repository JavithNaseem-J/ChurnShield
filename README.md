## Tele-Comm Churn Prediction

### 📌 Project Overview

This project aims to predict whether a telecommunications customer will churn or not churn based on various customer attributes and service details. The project leverages machine learning techniques to analyze customer data and provide actionable insights for reducing churn rates. It follows an end-to-end machine learning pipeline, including data ingestion, validation, transformation, model training, evaluation, and deployment with a front-end interface for user interaction.


flowchart TD
    subgraph "ML Pipeline"
        subgraph "Data Ingestion Stage"
            DI1["Data Ingestion (Data Folder)"]
            DI2["Data Ingestion (Component)"]
            DI3["Data Ingestion (Pipeline)"]
        end
        subgraph "Data Validation Stage"
            DV1["Data Validation (Artifacts)"]
            DV2["Data Validation (Schema)"]
            DV3["Data Validation (Component)"]
            DV4["Data Validation (Pipeline)"]
        end
        subgraph "Data Transformation Stage"
            DT1["Data Transformation (Artifacts)"]
            DT2["Data Transformation (Component)"]
            DT3["Data Transformation (Pipeline)"]
        end
        subgraph "Model Training Stage"
            MT1["Model Training (Artifacts)"]
            MT2["Model Training (Component)"]
            MT3["Model Training (Pipeline)"]
        end
        subgraph "Model Evaluation Stage"
            ME1["Model Evaluation (Artifacts)"]
            ME2["Model Evaluation (Component)"]
            ME3["Model Evaluation (Pipeline)"]
        end
    end

    DI3 -->|"stage"| DV1
    DV4 -->|"stage"| DT1
    DT3 -->|"stage"| MT1
    MT3 -->|"stage"| ME1
    MT1 -->|"trained_model"| WI1

    subgraph "Configuration & Control"
        Config1["Config Folder"]
        Config2["Params File"]
        Config3["Config Module"]
        Config4["Constants & Entities"]
    end

    Config3 ---|"feeds"| DI1

    subgraph "Web Interface"
        WI1["Flask App"]
        WI2["Templates"]
    end

    subgraph "Deployment & Automation"
        DA1["Docker"]
        DA2["CI/CD Workflow"]
    end

    WI1 -->|"containerized_in"| DA1
    DA1 -->|"triggered_by"| DA2
    DA2 -->|"deploys_to"| AWS["AWS (ECR & EC2)"]:::external

    subgraph "Research & Experimentation"
        RE1["Research Notebooks"]
    end

    %% Click Events
    click DI1 "https://github.com/javithnaseem-j/tele-com-customer-churn-prediction/tree/main/artifacts/data_ingestion"
    click DI2 "https://github.com/javithnaseem-j/tele-com-customer-churn-prediction/blob/main/src/mlproject/components/data_ingestion.py"
    click DI3 "https://github.com/javithnaseem-j/tele-com-customer-churn-prediction/blob/main/src/mlproject/pipeline/stage1_data_ingestion.py"

    click DV1 "https://github.com/javithnaseem-j/tele-com-customer-churn-prediction/tree/main/artifacts/data_validation"
    click DV2 "https://github.com/javithnaseem-j/tele-com-customer-churn-prediction/blob/main/schema.yaml"
    click DV3 "https://github.com/javithnaseem-j/tele-com-customer-churn-prediction/blob/main/src/mlproject/components/data_validation.py"
    click DV4 "https://github.com/javithnaseem-j/tele-com-customer-churn-prediction/blob/main/src/mlproject/pipeline/stage2_data_validation.py"

    click DT1 "https://github.com/javithnaseem-j/tele-com-customer-churn-prediction/tree/main/artifacts/data_transformation"
    click DT2 "https://github.com/javithnaseem-j/tele-com-customer-churn-prediction/blob/main/src/mlproject/components/data_transformation.py"
    click DT3 "https://github.com/javithnaseem-j/tele-com-customer-churn-prediction/blob/main/src/mlproject/pipeline/stage3_data_transformation.py"

    click MT1 "https://github.com/javithnaseem-j/tele-com-customer-churn-prediction/tree/main/artifacts/model_trainer"
    click MT2 "https://github.com/javithnaseem-j/tele-com-customer-churn-prediction/blob/main/src/mlproject/components/data_modeltraining.py"
    click MT3 "https://github.com/javithnaseem-j/tele-com-customer-churn-prediction/blob/main/src/mlproject/pipeline/stage4_modeltraining.py"

    click ME1 "https://github.com/javithnaseem-j/tele-com-customer-churn-prediction/tree/main/artifacts/model_evaluation"
    click ME2 "https://github.com/javithnaseem-j/tele-com-customer-churn-prediction/blob/main/src/mlproject/components/data_modelevaluation.py"
    click ME3 "https://github.com/javithnaseem-j/tele-com-customer-churn-prediction/blob/main/src/mlproject/pipeline/stage5_data_evalution.py"

    click Config1 "https://github.com/javithnaseem-j/tele-com-customer-churn-prediction/tree/main/config"
    click Config2 "https://github.com/javithnaseem-j/tele-com-customer-churn-prediction/blob/main/params.yaml"
    click Config3 "https://github.com/javithnaseem-j/tele-com-customer-churn-prediction/tree/main/src/mlproject/config"
    click Config4 "https://github.com/javithnaseem-j/tele-com-customer-churn-prediction/tree/main/src/mlproject/constants"

    click WI1 "https://github.com/javithnaseem-j/tele-com-customer-churn-prediction/blob/main/app.py"
    click WI2 "https://github.com/javithnaseem-j/tele-com-customer-churn-prediction/tree/main/templates"

    click DA1 "https://github.com/javithnaseem-j/tele-com-customer-churn-prediction/tree/main/Dockerfile"
    click DA2 "https://github.com/javithnaseem-j/tele-com-customer-churn-prediction/blob/main/.github/workflows/cicd.yaml"

    click RE1 "https://github.com/javithnaseem-j/tele-com-customer-churn-prediction/tree/main/research"

    %% Styles
    classDef pipeline fill:#AED6F1,stroke:#1F618D,stroke-width:2px;
    classDef config fill:#A9DFBF,stroke:#145A32,stroke-width:2px;
    classDef web fill:#F9E79F,stroke:#B7950B,stroke-width:2px;
    classDef deploy fill:#F5CBA7,stroke:#873600,stroke-width:2px;
    classDef external fill:#FADBD8,stroke:#922B21,stroke-width:2px;

    class DI1,DI2,DI3,DV1,DV2,DV3,DV4,DT1,DT2,DT3,MT1,MT2,MT3,ME1,ME2,ME3 pipeline;
    class Config1,Config2,Config3,Config4 config;
    class WI1,WI2 web;
    class DA1,DA2 deploy;
    class AWS,RE1 external;


### 📂 Dataset

The dataset used in this project, `Tele_Comm.csv`, contains the following key features:

- **Customer Demographics:** Gender, SeniorCitizen, Partner, Dependents
- **Service Information:** tenure (in months), PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
- **Contract Details:** Contract (Month-to-month, One year, Two year), PaperlessBilling, PaymentMethod
- **Billing Information:** MonthlyCharges, TotalCharges
- **Target Variable:** Churn (Yes/No)

### 🏷️ Project Workflow

#### **Data Ingestion:**
- The dataset (`Tele_Comm.csv`) is ingested and stored.

#### **Data Validation:**
- Validates the dataset schema and structure using `schema.yaml` to ensure data integrity.

#### **Data Transformation:**
- Preprocessed numerical features (tenure, MonthlyCharges, TotalCharges) with scaling and categorical features with label encoding.

#### **Model Training:**
- Trained a `RandomForestClassifier` on the preprocessed training data.
- Saved the trained model as `model.joblib`.

#### **Model Evaluation:**
- Evaluated the model on the test set, achieving the following metrics:
  - **Accuracy:** 85%
  - **Precision:** 84.74%
  - **Recall:** 85%
  - **F1-Score:** 85%

### **Front-End Development:**
- Developed a web interface using **Flask**, with `index.html` for input collection and `results.html` for displaying predictions.
- The interface allows users to input customer details and predict churn probability.

### **Deployment:**
- Containerized the project using **Docker**.
- Implemented a CI/CD pipeline for automated deployment via **GitHub Actions** (`cicd.yaml`).

---

## **Local Setup Steps**

### **1. Clone the Repository**
```bash
 git clone https://github.com/entbappy/Tele-Com-Customer-Churn-Prediction.git
 cd Tele-Com-Customer-Churn-Prediction-
```

### **2. Create a Conda Environment**
```bash
 conda create -n mlproj python=3.10 -y
 conda activate mlproj
```

### **3. Install Requirements**
```bash
 pip install -r requirements.txt
```

### **4. Run the Application**
```bash
 python app.py
```

---

## **AWS CI/CD Deployment with GitHub Actions**

### **1. Log in to AWS Console**
- Go to [AWS Console](https://aws.amazon.com/) and sign in.
- If you don’t have an account, create one and set up billing.

### **2. Create an IAM User for Deployment**
- Navigate to **IAM > Users > Add user** in the AWS Console.
- Set a username (e.g., `deployment-user`).
- Select **Programmatic access** as the access type.
- Attach the following policies:
  - `AmazonEC2ContainerRegistryFullAccess`: Grants full access to ECR.
  - `AmazonEC2FullAccess`: Grants full access to EC2.
- Review and create the user.
- Save the **Access Key ID** and **Secret Access Key** securely.

### **3. Create an ECR Repository**
- Go to **ECR > Repositories > Create repository** in the AWS Console.
- Select **Private** and name it (e.g., `mlproj`).
- Note the repository URI (e.g., `970547337635.dkr.ecr.ap-south-1.amazonaws.com/mlproj`).

### **4. Create an EC2 Instance**
- Go to **EC2 > Instances > Launch Instance** in the AWS Console.
- Choose an Ubuntu AMI (e.g., Ubuntu Server 20.04).
- Select an instance type (e.g., `t2.micro` for free tier eligibility).
- Configure a security group:
  - Allow inbound traffic on port 22 (SSH).
  - Allow the port your app uses (e.g., 5000).
- Launch the instance and download the key pair (e.g., `my-key.pem`).

### **5. Install Docker on EC2**
```bash
 sudo apt-get update -y
 sudo apt-get upgrade -y
 curl -fsSL https://get.docker.com -o get-docker.sh
 sudo sh get-docker.sh
 sudo usermod -aG docker ubuntu
 newgrp docker
```

### **6. Configure EC2 as a Self-Hosted Runner**
- Go to your GitHub repository > **Settings** > **Actions** > **Runners** > **New self-hosted runner**.
- Select the OS (**Linux**) and architecture (**X64**).
- Follow the provided instructions to:
  - Download the runner software.
  - Configure it with a token.
  - Start the runner on your EC2 instance.

### **7. Set Up GitHub Secrets**
- Go to your GitHub repository > **Settings** > **Secrets** > **New repository secret**.
- Add the following secrets:
  - `AWS_ACCESS_KEY_ID`: Your IAM user’s access key.
  - `AWS_SECRET_ACCESS_KEY`: Your IAM user’s secret key.
  - `AWS_REGION`: The region of your ECR (e.g., `ap-south-1`).
  - `AWS_ECR_LOGIN_URI`: The ECR login URI (e.g., `970547337635.dkr.ecr.ap-south-1.amazonaws.com`).
  - `ECR_REPOSITORY_NAME`: The name of your ECR repository (e.g., `mlproj`).



This project is licensed under the MIT License. See the LICENSE file for details.
