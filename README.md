# Cloud Native Streaming Prediction of Cryptocurrency Breakout vs Reversal Using ML

**Course:** CMPE 281 – Intelligent Cloud Platform  
**Team:** Johnny Nguyen (015849717), Jasper Nguyen (015176759)  
**Program:** Software Engineering

---

## Overview

This system predicts whether a cryptocurrency price anomaly will result in a **breakout** (price continues moving in the same direction) or a **reversal** (price returns to pre-anomaly levels). It uses a fully serverless AWS architecture with real-time data ingestion, SageMaker-trained ML models, and an automated alerting system.

**Supported symbols:** BTCUSDT, XRPUSDT  
**Data source:** Binance US public API (no API key required)  
**Models:** Logistic Regression, Random Forest, Gradient Boosting (breakout/reversal) + Linear Regression (price prediction)

---

## Prerequisites

- Python 3.12
- AWS account with admin access
- AWS CLI installed and configured
- Git

---

## Local Setup

```bash
git clone <your-repo>
cd cryptopredictor
python3 -m venv .venv
source .venv/bin/activate
pip install boto3 sagemaker requests pandas numpy scikit-learn joblib matplotlib python-dotenv pytest
```

Configure AWS CLI with your credentials:
```bash
aws configure
```
Enter your Access Key ID, Secret Access Key, region (`us-east-2`), and press Enter for output format.

---

## AWS Infrastructure Setup

All services are created in the **AWS Console** in the same region (`us-east-2`). Follow these steps in order.

### 1. S3 Bucket
1. Go to **S3** → **Create bucket**
2. Name: `ml-crypto-data`
3. Enable **versioning** and **server-side encryption (SSE-S3)**
4. Add a **lifecycle policy**: transition to Glacier after 90 days
5. Block all public access

### 2. DynamoDB Table
1. Go to **DynamoDB** → **Create table**
2. Table name: `crypto-predictions`
3. Partition key: `anomaly_id` (String)
4. Sort key: `timestamp` (String)
5. Capacity: **On-demand**

### 3. SQS Queues
**Main queue:**
1. Go to **SQS** → **Create queue**
2. Name: `crypto-event-queue`
3. Visibility timeout: `120` seconds
4. Message retention: `4` days
5. Receive message wait time: `10` seconds

**Dead letter queue:**
1. Create another queue named `crypto-processing-dlq`
2. Go back to `crypto-event-queue` → **Edit** → enable Dead-letter queue → select `crypto-processing-dlq` → Max receives: `3`

### 4. SNS Topic
1. Go to **SNS** → **Create topic**
2. Name: `crypto-alerts`, Type: Standard
3. Create a subscription with your email and confirm it

### 5. Secrets Manager
1. Go to **Secrets Manager** → **Store a new secret**
2. Type: Other → Key: `BINANCE_API_KEY`, Value: `free-no-key-needed`
3. Name: `crypto-project/api-keys`

### 6. IAM Roles
Create these roles in **IAM** → **Roles** → **Create role** → AWS service → Lambda:

**crypto-ingestion-lambda-role**
- Permissions: S3 PutObject, SQS SendMessage, Secrets Manager read, CloudWatch logs

**crypto-feature-eng-lambda-role**
- Permissions: SQS ReceiveMessage/DeleteMessage, S3 GetObject, Lambda InvokeFunction, CloudWatch logs

**crypto-inference-lambda-role**
- Permissions: S3 GetObject, DynamoDB PutItem, SNS Publish, CloudWatch logs

**crypto-dashboard-api-role**
- Permissions: DynamoDB Scan, CloudWatch logs

**crypto-sagemaker-role** (AWS service → SageMaker)
- Permissions: AmazonSageMakerFullAccess, AmazonS3FullAccess

### 7. CloudWatch
1. Go to **CloudWatch** → **Log groups** → create log groups for each Lambda with 14-day retention:
   - `/aws/lambda/crypto-ingestion`
   - `/aws/lambda/crypto-feature-engineering`
   - `/aws/lambda/crypto-inference`
   - `/aws/lambda/crypto-dashboard-api`
2. Create an error alarm on the ingestion Lambda that notifies via SNS

### 8. AWS Budget
1. Go to **Billing** → **Budgets** → **Create budget**
2. Set a monthly cost budget with an 80% alert threshold

---

## Lambda Layer (scikit-learn)

Run these commands in **AWS CloudShell** (the `>_` icon in the AWS Console top nav):

```bash
mkdir -p python
pip install scikit-learn==1.6.1 joblib numpy --no-cache-dir -t python/ --platform manylinux2014_x86_64 --python-version 3.12 --only-binary=:all:
find python -name "tests" -type d -exec rm -rf {} + 2>/dev/null
find python -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
zip -r sklearn-layer.zip python/
aws s3 cp sklearn-layer.zip s3://ml-crypto-data/sklearn-layer.zip
aws lambda publish-layer-version --layer-name sklearn-layer --content S3Bucket=ml-crypto-data,S3Key=sklearn-layer.zip --compatible-runtimes python3.12
```

Note the **Version** number in the output — you'll need it when creating the inference Lambda.

---

## Train Models with SageMaker

Run the training job locally (this launches training in the cloud):

```bash
python run.py
```

This uploads training data to S3 and starts a SageMaker Training Job. Monitor progress in the AWS Console under **SageMaker** → **Training** → **Training Jobs**. Takes 3-5 minutes.

After the job completes, extract and upload the models to S3:

```bash
mkdir -p sagemaker_output
aws s3 cp s3://ml-crypto-data/sagemaker/model-output/ sagemaker_output/ --recursive

mkdir -p extracted_models
tar -xzf sagemaker_output/<job-folder>/output/model.tar.gz -C extracted_models

aws s3 cp extracted_models/BTCUSDT/ s3://ml-crypto-data/model_artifacts/BTCUSDT/ --recursive
aws s3 cp extracted_models/XRPUSDT/ s3://ml-crypto-data/model_artifacts/XRPUSDT/ --recursive
```

Replace `<job-folder>` with the actual folder name (e.g. `sagemaker-scikit-learn-2026-04-26-18-35-32-676`).

---

## Deploy Lambda Functions

Create 4 Lambda functions in the **AWS Console** → **Lambda** → **Create function**. All use **Python 3.12**.

### Lambda 1: crypto-ingestion
- **File:** `lambda/ingestion/lambda_function.py`
- **Role:** `crypto-ingestion-lambda-role`
- **Timeout:** 30 seconds
- **Trigger:** EventBridge rule (every 1 hour)
- **Environment variables:**
  - `S3_BUCKET` = `ml-crypto-data`
  - `SQS_QUEUE_URL` = your SQS queue URL
  - `INTERVAL` = `1h`

### Lambda 2: crypto-feature-engineering
- **File:** `lambda/feature_engineering/lambda_function.py`
- **Role:** `crypto-feature-eng-lambda-role`
- **Timeout:** 60 seconds
- **Trigger:** SQS (`crypto-event-queue`, batch size 1)
- **Environment variables:**
  - `S3_BUCKET` = `ml-crypto-data`
  - `INFERENCE_FUNCTION_NAME` = `crypto-inference`
  - `ZSCORE_THRESHOLD` = `2.0`
  - `WINDOW_SHORT` = `24`
  - `WINDOW_LONG` = `168`

### Lambda 3: crypto-inference
- **File:** `lambda/inference/lambda_function.py`
- **Role:** `crypto-inference-lambda-role`
- **Timeout:** 60 seconds
- **Memory:** 512 MB
- **Layer:** `sklearn-layer` (attach the version you created)
- **Trigger:** Async invocation from feature engineering Lambda
- **Environment variables:**
  - `S3_BUCKET` = `ml-crypto-data`
  - `MODEL_PREFIX` = `model_artifacts`
  - `DYNAMODB_TABLE` = `crypto-predictions`
  - `SNS_TOPIC_ARN` = your SNS topic ARN
  - `CONFIDENCE_THRESHOLD` = `0.6`

### Lambda 4: crypto-dashboard-api
- **File:** `lambda/dashboard_api/lambda_function.py`
- **Role:** `crypto-dashboard-api-role`
- **Timeout:** 30 seconds
- **Trigger:** API Gateway
- **Environment variables:**
  - `DYNAMODB_TABLE` = `crypto-predictions`

---

## API Gateway Setup

1. Go to **API Gateway** → **Create API** → **HTTP API**
2. Add integration → Lambda → `crypto-dashboard-api`
3. Add routes: `GET /predictions` and `GET /stats`
4. Enable CORS: Allow-Origin `*`, Allow-Methods `GET OPTIONS`
5. Deploy and copy the **Invoke URL**
6. Update `dashboard.html` — replace `YOUR_API_GATEWAY_URL` with your actual URL

---

## EventBridge Setup

1. Go to **EventBridge** → **Rules** → **Create rule**
2. Name: `crypto-ingestion-schedule`
3. Rule type: Schedule → Rate: `1 hour`
4. Target: Lambda → `crypto-ingestion`

---

## Run the Dashboard

```bash
python3 -m http.server 8080
```

Open Chrome and go to `http://localhost:8080/dashboard.html`

---

## How the Pipeline Works

```
EventBridge (every hour)
    → crypto-ingestion Lambda
        → Fetches BTC + XRP candles from Binance US API
        → Saves raw JSON to S3 (raw/BTCUSDT/ and raw/XRPUSDT/)
        → Sends message to SQS

SQS message
    → crypto-feature-engineering Lambda
        → Reads 200 most recent files from S3
        → Computes 12 features per candle:
            log_return, volatility_short, volatility_long, volatility_ratio,
            volume_spike, ma_ratio, momentum_short, momentum_long,
            roc, return_zscore, daily_range, close_position
        → Detects anomalies (|z-score| >= 2.0)
        → If anomaly detected → invokes inference Lambda

Inference Lambda (only on anomalies)
    → Loads SageMaker-trained models from S3 (cached in memory)
    → Runs all 3 classification models → predicts breakout or reversal
    → Runs price prediction model → predicts next hour % price change
    → Saves results to DynamoDB
    → If confidence >= 60% → sends SNS email alert

Dashboard
    → Reads predictions from DynamoDB via API Gateway
    → Shows real-time predictions, features, model comparison, pipeline health
    → Auto-refreshes every 60 seconds
```

---

## Models

All models are **supervised learning** — trained on labeled historical data.

| Model | Type | Purpose |
|---|---|---|
| Logistic Regression | Supervised Classification | Baseline breakout/reversal predictor |
| Random Forest | Supervised Classification | Ensemble of 100 decision trees, handles complex patterns |
| Gradient Boosting | Supervised Classification | Iterative tree builder, usually most accurate |
| Linear Regression | Supervised Regression | Predicts next hour's % price change |

Models are trained per symbol (separate models for BTC and XRP) using SageMaker training jobs on `ml.m5.large` instances.

---

## Project Structure

```
cryptopredictor/
├── .venv/                          # Python virtual environment
├── data/                           # Local data files
├── docs/                           # Documentation
├── lambda/
│   ├── ingestion/
│   │   └── lambda_function.py      # Fetches data from Binance, saves to S3
│   ├── feature_engineering/
│   │   └── lambda_function.py      # Computes features, detects anomalies
│   ├── inference/
│   │   └── lambda_function.py      # Loads models, makes predictions
│   └── dashboard_api/
│       └── lambda_function.py      # Reads DynamoDB for dashboard
├── ml/
│   ├── sagemaker_train.py          # Training script (runs in SageMaker)
│   └── ml_pipeline.py             # Local training fallback
├── model_artifacts/                # Local copy of trained models
│   ├── BTCUSDT/
│   └── XRPUSDT/
├── dashboard.html                  # Frontend dashboard
├── raw_crypto_data_BTCUSDT.csv    # Training data
├── raw_crypto_data_XRPUSDT.csv    # Training data
├── run.py                         # Launches SageMaker training job
└── README.md
```

---

## Security

- IAM least-privilege roles for each Lambda
- Data encrypted at rest (S3 SSE-S3, DynamoDB default encryption)
- Data encrypted in transit (HTTPS/TLS for all AWS service communication)
- API keys stored in AWS Secrets Manager
- CloudWatch monitors all Lambda errors and latency
- AWS Budget alerts prevent unexpected costs

---

## Cost Optimization

- Serverless architecture — pay only when code runs
- ML inference only triggers on anomalies, not every candle
- S3 lifecycle policies move old data to Glacier after 90 days
- DynamoDB on-demand pricing — no provisioned capacity costs
- CloudWatch log retention set to 14 days