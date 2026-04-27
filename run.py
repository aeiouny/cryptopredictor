"""
Launch SageMaker Training Job
Run this locally to kick off model training in the cloud.

Setup:
  pip install sagemaker boto3
  
Usage:
  python launch_sagemaker_training.py
"""

import boto3
import sagemaker
from sagemaker.sklearn.estimator import SKLearn
import os

# ===================== CONFIG =====================
# Update these to match your AWS setup

S3_BUCKET = 'ml-crypto-data'
SAGEMAKER_ROLE = 'arn:aws:iam::733796381367:role/crypto-sagemaker-role'
REGION = 'us-east-2'  # Must match your other AWS services

# ===================== SETUP =====================

boto_session = boto3.Session(region_name=REGION)
sagemaker_session = sagemaker.Session(boto_session=boto_session)

# ===================== UPLOAD TRAINING DATA =====================
# Upload your local CSV files to S3 so SageMaker can access them

print("Uploading training data to S3...")

s3_client = boto3.client('s3', region_name=REGION)

training_files = [
    'raw_crypto_data_BTCUSDT.csv',
    'raw_crypto_data_XRPUSDT.csv'
]

for filename in training_files:
    if os.path.exists(filename):
        s3_key = f'sagemaker/training-data/{filename}'
        s3_client.upload_file(filename, S3_BUCKET, s3_key)
        print(f"Uploaded {filename} to s3://{S3_BUCKET}/{s3_key}")
    else:
        print(f"WARNING: {filename} not found locally. Run ml_pipeline.py first to generate it.")

training_data_uri = f's3://{S3_BUCKET}/sagemaker/training-data/'
print(f"Training data location: {training_data_uri}")

# ===================== CREATE ESTIMATOR =====================
# SKLearn estimator tells SageMaker which container to use and how to run training

print("\nCreating SageMaker estimator...")

sklearn_estimator = SKLearn(
    entry_point='sagemaker_train.py',       # Your training script
    role=SAGEMAKER_ROLE,                     # IAM role we created
    instance_type='ml.m5.large',            # Training instance (cheap, plenty for our data)
    instance_count=1,
    framework_version='1.2-1',              # sklearn version in the container
    py_version='py3',
    sagemaker_session=sagemaker_session,
    output_path=f's3://{S3_BUCKET}/sagemaker/model-output/',
    
    # Hyperparameters passed to the training script
    hyperparameters={
        'zscore-threshold': 2.0,
        'breakout-threshold': 0.02,
        'lookforward-window': 5,
        'window-short': 7,
        'window-long': 30,
    }
)

# ===================== LAUNCH TRAINING JOB =====================

print("\nLaunching SageMaker training job...")
print("This will take 3-5 minutes. You can monitor progress in the AWS Console.")
print("Go to: SageMaker > Training > Training Jobs")

sklearn_estimator.fit(
    {'training': training_data_uri},
    wait=True,      # Wait for job to complete
    logs=True       # Stream logs to terminal
)

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print(f"Model artifacts saved to: s3://{S3_BUCKET}/sagemaker/model-output/")
print("\nNext steps:")
print("1. The model files are in S3 under sagemaker/model-output/")
print("2. Copy them to model_artifacts/BTCUSDT/ and model_artifacts/XRPUSDT/")
print("3. Your inference Lambda will pick them up automatically")