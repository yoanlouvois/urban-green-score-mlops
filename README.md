# Urban Green Score - MLOps Pipeline

## Project Overview

This project implements a production-grade MLOps pipeline designed to automate the calculation of an Urban Green Score from satellite imagery. By leveraging deep learning and cloud infrastructure, the system transforms raw geospatial data into actionable environmental insights.
The core workflow includes:
- Semantic Segmentation: Utilizing a U-Net architecture to identify land cover.
- Green Scoring: Computing an environmental index based on detected vegetation (forest, agriculture, etc.).
- Automated Lifecycle: From data preprocessing to serverless inference deployment.
- The final model is accessible through two interfaces: a Streamlit app and a custom web front-end (FastAPI + HTML/JS). Both let users upload satellite images and get the Green Score instantly, using either the cloud endpoint or the local model: 

<table align="center">
  <tr>
    <th align="center"> Visualisation Streamlit</th>
    <th align="center"> Front-end FastAPI</th>
  </tr>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/667bfe54-9d4a-49d8-87b8-a6ea8d2ae855" width="100%" />
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/c778b010-0b2c-4d0e-89ce-0cf08cfff58a" width="100%" />
    </td>
  </tr>
</table>


---

## MLOps Pipeline

The architecture is built for scalability and reproducibility, using industry-standard tools:

- **PyTorch** : semantic segmentation model (**U-Net**)
- **Docker** : reproducible training/inference environments
- **AWS**
  - S3 : dataset & model artifact storage
  - ECR : container registry
  - SageMaker : processing, training, evaluation, deployment
  - CloudWatch : logs & monitoring
  - Lambda : serverless inference proxy
  - API Gateway : HTTP endpoint exposure
- **Terraform** : infrastructure as code
- **Github Actions** : CI/CD pipelines

The following diagram illustrates the system architecture : 


<img width="1530" height="566" alt="Diagramme sans nom (11)" src="https://github.com/user-attachments/assets/1c66d635-7819-43e6-b22f-e521035d0c03" />


---

## Setup

### 1. Prerequisites

To run this project, you will need:

- An AWS account with permissions to use SageMaker, Lambda, API Gateway, IAM, CloudWatch, ECR, and S3
- Docker installed on your local machine
- Terraform installed for infrastructure provisioning
- Python 3.11+
- AWS CLI installed and configured
- GitHub account (optional, for CI/CD workflows)

### 2. Clone Repository

First, clone the repository and navigate into the project folder:
```bash
git clone https://github.com/<your-username>/Urban-Green-Score-MLops.git
cd Urban-Green-Score-MLops
```

### 3. Configure AWS CLI

Authenticate your AWS account locally:
```bash
aws configure
```

### 4. Provision Infrastructure

Navigate to the Terraform folder:
```bash
cd terraform
```
Update the `terraform.tfvars` file with your configuration.

Then initialize Terraform:
```bash
terraform init
```
Preview infrastructure changes:
```bash
terraform plan
```
Deploy infrastructure:
```bash
terraform apply
```
This will provision the following AWS resources:

- S3 bucket for datasets and model artifacts
- ECR repository for Docker images
- IAM roles and policies
- SageMaker model + endpoint configuration
- SageMaker endpoint (optional via deploy_endpoint flag)
- Lambda function
- API Gateway
- CloudWatch dashboard

### 5. Configure Environment Variables

Create a `.env` file at the root of the project using the provided template:
```bash
cp .env.example .env
```
Then configure your AWS and project variables inside `.env`.

### 6. Setup Local Python Environment

Create and activate a local virtual environment to run the SageMaker scripts, Streamlit app, and utility scripts locally.

Create the virtual environment:

```bash
python -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements-local.txt
```
This installs SageMaker SDK, boto3, Streamlit, pytest and other local development utilities

### 7. Upload Dataset

This project can work with any satellite semantic segmentation dataset containing images and masks.

For this implementation, I personally used the **LoveDA Satellite Images Semantic Segmentation** dataset:  
https://www.kaggle.com/datasets/alienxc137/loveda-satellite-images-semantic-segmentation

Download the dataset manually and place it inside the `data/raw/` folder.

Expected structure:
```bash
data/
  raw/
    train/
      images/
      masks/
    val/
      images/
      masks/
    test/
      images/
```
Then upload the raw dataset to your S3 bucket:
```bash
aws s3 cp data/raw s3://your-bucket-name/data/raw --recursive
```

### 8. Docker Authentication

Authenticate Docker with your private ECR repository:
```bash
aws ecr get-login-password --region eu-west-3 | docker login --username AWS --password-stdin <your-account-id>.dkr.ecr.eu-west-3.amazonaws.com
```

Build the Docker image used by SageMaker:

```bash
docker build --platform linux/amd64 --provenance=false -t urban-green-score .
```

Tag the image:

```bash
docker tag urban-green-score:latest 147914447581.dkr.ecr.eu-west-3.amazonaws.com/urban-green-score:latest
```

Push the image to AWS ECR:

```bash
docker push 147914447581.dkr.ecr.eu-west-3.amazonaws.com/urban-green-score:latest
```

---
&nbsp;
## Pipelines

### 1. Data Processing

Run SageMaker Processing job:

```bash
python src/scripts/run_processing.py
```

This stepdownloads raw data
- preprocesses satellite images
- resizes images
- prepares masks
- uploads processed dataset to S3


### 2. Model Training

Launch SageMaker Training job:

```bash
python src/scripts/run_training.py
```

This step:

- trains the U-Net segmentation model
- stores model artifacts in the S3 bucket
- logs training metrics in CloudWatch


### 3. Model Evaluation

Launch SageMaker Evaluation job:

```bash
python src/scripts/run_evaluation.py
```

This step:

- evaluates model performance
- computes IoU / accuracy metrics
- saves evaluation artifacts

### 4. Model Registry

Register the trained model:

```bash
python src/scripts/model_registry/register_model.py
```

This creates a new version inside SageMaker Model Registry.


### 5. Model Deployment

If you want to deploy the SageMaker endpoint, set `deploy_endpoint=true` in `terraform.tfvars` or run:

```bash
terraform apply -var="deploy_endpoint=true"
```

This creates:

- SageMaker Model
- Endpoint Configuration
- Real-time Endpoint

Use deploy_endpoint=false when you want to avoid endpoint costs.

### 6. Test Endpoint Inference

Two interfaces are available to test the model. Both support two inference modes:
- **Cloud API**: sends the image to API Gateway → Lambda → SageMaker endpoint (the endpoint must be `InService`)
- **Local model**: runs inference directly with `artifacts/models/best_model.pth`

**Option A: Web front-end (FastAPI)**

```bash
uvicorn server:app --app-dir src/app/web --reload
```
Then open http://localhost:8000.

Features: drag & drop upload, Green Score gauge, side-by-side or overlay view of the segmentation mask (with adjustable opacity) and per-class proportions.

The API Gateway URL and model path can be overridden with the `API_GATEWAY_URL` and `LOCAL_MODEL_PATH` environment variables.

**Option B: Streamlit app**

```bash
streamlit run src/app/streamlit_app.py
```
### 7. Monitoring

This project uses Amazon CloudWatch to monitor deployed services.

Tracked metrics:

- Lambda invocations/errors
- API Gateway requests/errors
- SageMaker endpoint logs

A CloudWatch dashboard is automatically provisioned with Terraform for quick monitoring and debugging.


### 8. Cleanup 

If you want to delete the SageMaker endpoint, set `deploy_endpoint=false` in `terraform.tfvars` or run:

```bash
terraform apply -var="deploy_endpoint=false"
```

---
&nbsp;
## CI/CD Pipeline

This project uses GitHub Actions to automate testing, Docker deployment, and infrastructure validation.

Required GitHub Secrets:

- AWS_ACCESS_KEY_ID
- AWS_SECRET_ACCESS_KEY
- AWS_REGION
- AWS_ACCOUNT_ID
- ECR_REPOSITORY

Pipeline Workflows:

- CI (`ci.yml`) : executes lint checks, runs unit tests and validates Docker build
- Deploy ECR (`deploy-ecr.yml`) : builds Docker image, pushes image to Amazon ECR
- Terraform Check (`terraform.yml`) : executes terraform fmt, runs terraform validate and runs terraform plan

Every push or pull request automatically triggers the corresponding workflow.

You can monitor workflow execution in the GitHub Actions dashboard.

---

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

