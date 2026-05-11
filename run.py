import boto3
import sagemaker
from sagemaker.sklearn.estimator import SKLearn
import requests
import pandas as pd
import os
import tarfile
import tempfile

S3_BUCKET = 'ml-crypto-data'
SAGEMAKER_ROLE = 'arn:aws:iam::733796381367:role/crypto-sagemaker-role'
REGION = 'us-east-2'

boto_session = boto3.Session(region_name=REGION)
sagemaker_session = sagemaker.Session(boto_session=boto_session)
s3_client = boto3.client('s3', region_name=REGION)

# Step 1: Fetch training data from Binance
print("Fetching training data from Binance...")
for symbol in ['BTCUSDT', 'XRPUSDT']:
    all_candles = []
    # Fetch 1000 candles at a time going back 1 year
    end_time = None
    for _ in range(9):  # 9 batches x 1000 = 9000 candles (~1 year)
        params = f'symbol={symbol}&interval=1h&limit=1000'
        if end_time:
            params += f'&endTime={end_time}'
        data = requests.get(f'https://api.binance.us/api/v3/klines?{params}').json()
        if not data:
            break
        all_candles = data + all_candles
        end_time = data[0][0] - 1  # go further back in time

    df = pd.DataFrame(all_candles, columns=[
        'timestamp','open','high','low','close','volume',
        'close_time','quote_volume','trades','taker_buy_base','taker_buy_quote','ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.drop_duplicates('timestamp').sort_values('timestamp')
    for col in ['open','high','low','close','volume']:
        df[col] = df[col].astype(float)
    filename = f'raw_crypto_data_{symbol}.csv'
    df[['timestamp','open','high','low','close','volume']].to_csv(filename, index=False)
    print(f"Saved {filename} with {len(df)} rows")

# Step 2: Upload CSVs to S3
print("\nUploading training data to S3...")
for symbol in ['BTCUSDT', 'XRPUSDT']:
    filename = f'raw_crypto_data_{symbol}.csv'
    s3_key = f'sagemaker/training-data/{filename}'
    s3_client.upload_file(filename, S3_BUCKET, s3_key)
    print(f"Uploaded {filename}")

training_data_uri = f's3://{S3_BUCKET}/sagemaker/training-data/'

# Step 3: Launch SageMaker training job
print("\nLaunching SageMaker training job...")
sklearn_estimator = SKLearn(
    entry_point='sagemaker_train.py',
    role=SAGEMAKER_ROLE,
    instance_type='ml.m5.large',
    instance_count=1,
    framework_version='1.2-1',
    py_version='py3',
    sagemaker_session=sagemaker_session,
    output_path=f's3://{S3_BUCKET}/sagemaker/model-output/',
    hyperparameters={
        'zscore-threshold': 2.0,
        'breakout-threshold': 0.005,
        'lookforward-window': 5,
        'window-short': 7,
        'window-long': 30,
    }
)

sklearn_estimator.fit({'training': training_data_uri}, wait=True, logs=True)

# Step 4: Auto-extract and upload models to S3
print("\nExtracting and uploading models to S3...")
job_name = sklearn_estimator.latest_training_job.name
tar_key = f'sagemaker/model-output/{job_name}/output/model.tar.gz'

with tempfile.TemporaryDirectory() as tmp:
    local_tar = os.path.join(tmp, 'model.tar.gz')
    s3_client.download_file(S3_BUCKET, tar_key, local_tar)
    print(f"Downloaded model.tar.gz")

    with tarfile.open(local_tar, 'r:gz') as tar:
        tar.extractall(tmp)
    print("Extracted models")

    for symbol in ['BTCUSDT', 'XRPUSDT']:
        symbol_dir = os.path.join(tmp, symbol)
        if not os.path.exists(symbol_dir):
            print(f"Warning: {symbol} folder not found in extracted models")
            continue
        for filename in os.listdir(symbol_dir):
            local_path = os.path.join(symbol_dir, filename)
            s3_key = f'model_artifacts/{symbol}/{filename}'
            s3_client.upload_file(local_path, S3_BUCKET, s3_key)
            print(f"Uploaded {symbol}/{filename}")

print("\nDONE — Models are live in S3 and ready for inference.")
print("Remember to clear the Lambda cache by updating any env variable on crypto-inference.")