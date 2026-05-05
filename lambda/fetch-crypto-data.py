import json
import boto3
import os
from datetime import datetime, timezone
from urllib.request import urlopen, Request
from urllib.parse import urlencode

s3 = boto3.client('s3')
sqs = boto3.client('sqs')
S3_BUCKET = os.environ.get('S3_BUCKET')
SQS_QUEUE_URL = os.environ.get('SQS_QUEUE_URL')
INTERVAL = os.environ.get('INTERVAL', '1h')
SYMBOLS = ['BTCUSDT', 'XRPUSDT']


def fetch_binance_data(symbol, interval, limit=500):
    base_url = "https://api.binance.us/api/v3/klines"
    params = urlencode({"symbol": symbol, "interval": interval, "limit": limit})
    req = Request(f"{base_url}?{params}", headers={"User-Agent": "CryptoLambda/1.0"})
    
    with urlopen(req, timeout=10) as response:
        raw_data = json.loads(response.read().decode())
    
    return [{
        "timestamp": c[0], "open": float(c[1]), "high": float(c[2]),
        "low": float(c[3]), "close": float(c[4]), "volume": float(c[5]),
        "close_time": c[6], "num_trades": int(c[8]),
        "symbol": symbol, "interval": interval
    } for c in raw_data]


def save_to_s3(candles, symbol):
    now = datetime.now(timezone.utc)
    s3_key = f"raw/{symbol}/{now.strftime('%Y/%m/%d/%Y%m%d_%H%M%S')}.json"
    s3.put_object(
        Bucket=S3_BUCKET, Key=s3_key,
        Body=json.dumps({"symbol": symbol, "fetched_at": now.isoformat(), "candles": candles}),
        ContentType="application/json"
    )
    return s3_key


def send_sqs_message(s3_key, symbol, candles):
    sqs.send_message(
        QueueUrl=SQS_QUEUE_URL,
        MessageBody=json.dumps({
            "source": "ingestion-lambda", "s3_bucket": S3_BUCKET,
            "s3_key": s3_key, "symbol": symbol,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "latest_candle": candles[-1] if candles else None,
            "num_candles": len(candles)
        })
    )


def lambda_handler(event, context):
    results = []
    for symbol in SYMBOLS:
        try:
            candles = fetch_binance_data(symbol, INTERVAL)
            s3_key = save_to_s3(candles, symbol)
            send_sqs_message(s3_key, symbol, candles)
            results.append({"symbol": symbol, "status": "success", "s3_key": s3_key})
        except Exception as e:
            results.append({"symbol": symbol, "status": "error", "error": str(e)})
    
    return {"statusCode": 200, "body": json.dumps({"results": results})}