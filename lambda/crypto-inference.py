import json
import boto3
import os
import uuid
from datetime import datetime, timezone
import tempfile

s3 = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')
sns = boto3.client('sns')

S3_BUCKET = os.environ.get('S3_BUCKET')
MODEL_PREFIX = os.environ.get('MODEL_PREFIX', 'model_artifacts')
DYNAMODB_TABLE = os.environ.get('DYNAMODB_TABLE')
SNS_TOPIC_ARN = os.environ.get('SNS_TOPIC_ARN')
CONFIDENCE_THRESHOLD = float(os.environ.get('CONFIDENCE_THRESHOLD', '0.6'))

cached_models = {}


def load_model_from_s3(symbol):
    global cached_models
    if symbol in cached_models:
        return cached_models[symbol]

    import joblib
    tmp_dir = tempfile.mkdtemp()
    prefix = f"{MODEL_PREFIX}/{symbol}"

    models = {}
    for name in ['logistic_regression', 'random_forest', 'gradient_boosting']:
        try:
            path = os.path.join(tmp_dir, f"model_{name}.joblib")
            s3.download_file(S3_BUCKET, f"{prefix}/model_{name}.joblib", path)
            models[name] = joblib.load(path)
        except:
            continue

    if not models:
        path = os.path.join(tmp_dir, "model.joblib")
        s3.download_file(S3_BUCKET, f"{prefix}/model.joblib", path)
        models['best'] = joblib.load(path)

    price_model = None
    try:
        path = os.path.join(tmp_dir, "model_price_prediction.joblib")
        s3.download_file(S3_BUCKET, f"{prefix}/model_price_prediction.joblib", path)
        price_model = joblib.load(path)
    except:
        price_model = None

    scaler_path = os.path.join(tmp_dir, "scaler.joblib")
    s3.download_file(S3_BUCKET, f"{prefix}/scaler.joblib", scaler_path)
    scaler = joblib.load(scaler_path)

    config_path = os.path.join(tmp_dir, "model_config.json")
    s3.download_file(S3_BUCKET, f"{prefix}/model_config.json", config_path)
    with open(config_path, 'r') as f:
        config = json.load(f)

    cached_models[symbol] = (models, scaler, config, price_model)
    return models, scaler, config, price_model


def predict(features, models, scaler, config, price_model):
    import numpy as np

    feature_columns = config['feature_columns']
    best_model_name = config.get('best_model_name', 'best')

    X = np.array([features.get(col, 0) for col in feature_columns]).reshape(1, -1)
    X_scaled = scaler.transform(X)

    all_predictions = {}
    for name, model in models.items():
        pred = model.predict(X_scaled)[0]
        proba = model.predict_proba(X_scaled)[0]
        all_predictions[name] = {
            'prediction': 'breakout' if pred == 1 else 'reversal',
            'label': int(pred),
            'confidence': round(float(proba.max()), 4),
            'breakout_probability': round(float(proba[1]), 4) if len(proba) > 1 else 0,
            'reversal_probability': round(float(proba[0]), 4) if len(proba) > 1 else 0
        }

    primary = all_predictions.get(best_model_name, list(all_predictions.values())[0])

    predicted_pct_change = None
    if price_model is not None:
        try:
            predicted_pct_change = round(float(price_model.predict(X_scaled)[0]), 6)
        except:
            predicted_pct_change = None

    return {
        'prediction': primary['prediction'],
        'label': primary['label'],
        'confidence': primary['confidence'],
        'breakout_probability': primary['breakout_probability'],
        'reversal_probability': primary['reversal_probability'],
        'best_model': best_model_name,
        'all_model_predictions': all_predictions,
        'predicted_pct_change': predicted_pct_change
    }


def save_to_dynamodb(prediction_result, features, candle, anomaly_direction):
    table = dynamodb.Table(DYNAMODB_TABLE)
    anomaly_id = str(uuid.uuid4())
    timestamp = str(int(datetime.now(timezone.utc).timestamp()))

    close_price = candle.get('close', 0)
    pct_change = prediction_result.get('predicted_pct_change')
    predicted_price = None
    if pct_change is not None and close_price > 0:
        predicted_price = round(close_price * (1 + pct_change), 2)

    table.put_item(Item={
        'anomaly_id': anomaly_id,
        'timestamp': timestamp,
        'datetime': datetime.now(timezone.utc).isoformat(),
        'symbol': candle.get('symbol', 'BTCUSDT'),
        'prediction': prediction_result['prediction'],
        'label': prediction_result['label'],
        'confidence': str(prediction_result['confidence']),
        'breakout_probability': str(prediction_result['breakout_probability']),
        'reversal_probability': str(prediction_result['reversal_probability']),
        'best_model': prediction_result.get('best_model', 'unknown'),
        'all_model_predictions': json.dumps(prediction_result.get('all_model_predictions', {})),
        'predicted_pct_change': str(pct_change) if pct_change is not None else 'N/A',
        'predicted_price': str(predicted_price) if predicted_price is not None else 'N/A',
        'anomaly_direction': anomaly_direction,
        'return_zscore': str(features.get('return_zscore', 0)),
        'close_price': str(candle.get('close', 0)),
        'volume': str(candle.get('volume', 0)),
        'features': json.dumps(features)
    })

    return anomaly_id


def send_alert(prediction_result, candle, anomaly_direction, anomaly_id):
    if prediction_result['confidence'] < CONFIDENCE_THRESHOLD:
        return

    direction_str = "UP ↑" if anomaly_direction > 0 else "DOWN ↓"
    prediction = prediction_result['prediction'].upper()
    confidence = prediction_result['confidence']
    pct_change = prediction_result.get('predicted_pct_change')
    close_price = candle.get('close', 0)

    price_line = ""
    if pct_change is not None and close_price > 0:
        predicted_price = round(close_price * (1 + pct_change), 2)
        price_line = f"Predicted Price Change: {pct_change*100:+.3f}%\nPredicted Next Price: ${predicted_price:,.2f}\n"

    sns.publish(
        TopicArn=SNS_TOPIC_ARN,
        Subject=f"Crypto Alert: {prediction} ({confidence:.0%} confidence)"[:100],
        Message=(
            f"CRYPTO ANOMALY DETECTED\n"
            f"Symbol: {candle.get('symbol', 'BTCUSDT')}\n"
            f"Price: ${close_price:,.2f}\n"
            f"Direction: {direction_str}\n"
            f"Prediction: {prediction}\n"
            f"Confidence: {confidence:.1%}\n"
            f"{price_line}"
            f"Anomaly ID: {anomaly_id}\n"
            f"Time: {datetime.now(timezone.utc).isoformat()}"
        )
    )


def lambda_handler(event, context):
    try:
        features = event.get('features', {})
        candle = event.get('candle', {})
        anomaly_direction = event.get('anomaly_direction', 0)
        symbol = candle.get('symbol', 'BTCUSDT')

        if not features:
            return {"statusCode": 400, "body": "No features provided"}

        models, scaler, config, price_model = load_model_from_s3(symbol)

        prediction_result = predict(features, models, scaler, config, price_model)
        prediction_result['return_zscore'] = features.get('return_zscore', 0)

        anomaly_id = save_to_dynamodb(prediction_result, features, candle, anomaly_direction)
        send_alert(prediction_result, candle, anomaly_direction, anomaly_id)

        return {
            "statusCode": 200,
            "body": json.dumps({
                "anomaly_id": anomaly_id,
                "prediction": prediction_result['prediction'],
                "confidence": prediction_result['confidence'],
                "predicted_pct_change": prediction_result.get('predicted_pct_change')
            })
        }

    except Exception as e:
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}