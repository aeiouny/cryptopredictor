"""
=============================================================================
LAMBDA 3: INFERENCE SERVICE
=============================================================================
Triggered by: Feature Engineering Lambda (async invocation)
What it does:
  1. Loads the trained ML model from S3 (cached after first load)
  2. Scales the features using the saved scaler
  3. Predicts breakout vs reversal with confidence score
  4. Saves the prediction to DynamoDB
  5. If confidence is high enough, sends an alert via SNS

AWS Setup needed:
  - IAM role with: S3 GetObject, DynamoDB PutItem, SNS Publish, CloudWatch logs
  - Environment variables: S3_BUCKET, MODEL_PREFIX, DYNAMODB_TABLE, SNS_TOPIC_ARN, CONFIDENCE_THRESHOLD
  - Timeout: 60 seconds
  - Memory: 512 MB (model loading needs more memory)
  - Runtime: Python 3.12
  
IMPORTANT — Lambda Layer:
  This Lambda needs scikit-learn, numpy, and joblib. These aren't available 
  by default in Lambda. You have two options:
  
  Option A (recommended): Create a Lambda Layer with these packages.
    See the deployment guide for instructions.
    
  Option B: Use a container image instead of a zip deployment.
  
  For a quick demo, you can also use the "lightweight" prediction approach 
  at the bottom that doesn't need scikit-learn (but is less accurate).
=============================================================================
"""

import json
import boto3
import os
import uuid
from datetime import datetime, timezone
import tempfile

# AWS clients
s3 = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')
sns = boto3.client('sns')

# Environment variables
S3_BUCKET = os.environ.get('S3_BUCKET', 'YOUR-BUCKET-NAME')
MODEL_PREFIX = os.environ.get('MODEL_PREFIX', 'model_artifacts')
DYNAMODB_TABLE = os.environ.get('DYNAMODB_TABLE', 'crypto-predictions')
SNS_TOPIC_ARN = os.environ.get('SNS_TOPIC_ARN', 'YOUR-SNS-TOPIC-ARN')
CONFIDENCE_THRESHOLD = float(os.environ.get('CONFIDENCE_THRESHOLD', '0.6'))

# Cache model in memory between invocations (Lambda reuses containers)
cached_model = None
cached_scaler = None
cached_config = None


def load_model_from_s3():
    """
    Download and load model artifacts from S3.
    
    Caches the model in memory so we don't re-download on every invocation.
    Lambda keeps the container warm between calls, so this saves time.
    """
    
    global cached_model, cached_scaler, cached_config
    
    if cached_model is not None:
        print("Using cached model")
        return cached_model, cached_scaler, cached_config
    
    print("Loading model from S3...")
    
    # We need joblib and sklearn — these must be in a Lambda Layer
    import joblib
    
    tmp_dir = tempfile.mkdtemp()
    
    # Download model
    model_path = os.path.join(tmp_dir, "model.joblib")
    s3.download_file(S3_BUCKET, f"{MODEL_PREFIX}/model.joblib", model_path)
    model = joblib.load(model_path)
    print("Model loaded")
    
    # Download scaler
    scaler_path = os.path.join(tmp_dir, "scaler.joblib")
    s3.download_file(S3_BUCKET, f"{MODEL_PREFIX}/scaler.joblib", scaler_path)
    scaler = joblib.load(scaler_path)
    print("Scaler loaded")
    
    # Download config
    config_path = os.path.join(tmp_dir, "model_config.json")
    s3.download_file(S3_BUCKET, f"{MODEL_PREFIX}/model_config.json", config_path)
    with open(config_path, 'r') as f:
        config = json.load(f)
    print("Config loaded")
    
    # Cache for next invocation
    cached_model = model
    cached_scaler = scaler
    cached_config = config
    
    return model, scaler, config


def predict(features, model, scaler, config):
    """
    Run the model prediction.
    
    Parameters:
    -----------
    features : dict - feature values from the feature engineering Lambda
    model : trained sklearn model
    scaler : fitted StandardScaler
    config : dict with feature_columns list
    
    Returns:
    --------
    dict with prediction label and confidence score
    """
    
    import numpy as np
    
    feature_columns = config['feature_columns']
    
    # Arrange features in the correct order
    feature_values = [features.get(col, 0) for col in feature_columns]
    X = np.array(feature_values).reshape(1, -1)
    
    # Scale features (same scaling as training)
    X_scaled = scaler.transform(X)
    
    # Predict
    prediction = model.predict(X_scaled)[0]
    probabilities = model.predict_proba(X_scaled)[0]
    confidence = float(probabilities.max())
    
    return {
        'prediction': 'breakout' if prediction == 1 else 'reversal',
        'label': int(prediction),
        'confidence': round(confidence, 4),
        'breakout_probability': round(float(probabilities[1]), 4) if len(probabilities) > 1 else 0,
        'reversal_probability': round(float(probabilities[0]), 4) if len(probabilities) > 1 else 0
    }


def save_to_dynamodb(prediction_result, features, candle, anomaly_direction):
    """
    Save the prediction to DynamoDB for evaluation and dashboard.
    """
    
    table = dynamodb.Table(DYNAMODB_TABLE)
    
    # Create a unique ID for this anomaly
    anomaly_id = str(uuid.uuid4())
    
    # Current timestamp as a number (for sort key)
    timestamp = int(datetime.now(timezone.utc).timestamp())
    
    item = {
        'anomaly_id': anomaly_id,
        'timestamp': timestamp,
        'datetime': datetime.now(timezone.utc).isoformat(),
        'symbol': candle.get('symbol', 'BTCUSDT'),
        'prediction': prediction_result['prediction'],
        'label': prediction_result['label'],
        'confidence': str(prediction_result['confidence']),
        'breakout_probability': str(prediction_result['breakout_probability']),
        'reversal_probability': str(prediction_result['reversal_probability']),
        'anomaly_direction': anomaly_direction,
        'return_zscore': str(features.get('return_zscore', 0)),
        'close_price': str(candle.get('close', 0)),
        'volume': str(candle.get('volume', 0)),
        'features': json.dumps(features)
    }
    
    table.put_item(Item=item)
    
    print(f"Saved to DynamoDB: {anomaly_id}")
    
    return anomaly_id


def send_alert(prediction_result, candle, anomaly_direction, anomaly_id):
    """
    Send an SNS alert to notify the user about the prediction.
    
    Only sends if confidence exceeds the threshold — this is the 
    "fewer, higher-quality alerts" goal from your proposal.
    """
    
    confidence = prediction_result['confidence']
    
    if confidence < CONFIDENCE_THRESHOLD:
        print(f"Confidence {confidence:.1%} below threshold {CONFIDENCE_THRESHOLD:.1%}. "
              f"No alert sent.")
        return
    
    direction_str = "UP ↑" if anomaly_direction > 0 else "DOWN ↓"
    prediction = prediction_result['prediction'].upper()
    
    subject = f"Crypto Alert: {prediction} predicted ({confidence:.0%} confidence)"
    
    message = (
        f"🚨 CRYPTO ANOMALY DETECTED\n"
        f"{'='*40}\n\n"
        f"Symbol: {candle.get('symbol', 'BTCUSDT')}\n"
        f"Price: ${candle.get('close', 'N/A'):,.2f}\n"
        f"Anomaly Direction: {direction_str}\n"
        f"Z-Score: {prediction_result.get('return_zscore', 'N/A')}\n\n"
        f"PREDICTION: {prediction}\n"
        f"Confidence: {confidence:.1%}\n"
        f"Breakout Probability: {prediction_result['breakout_probability']:.1%}\n"
        f"Reversal Probability: {prediction_result['reversal_probability']:.1%}\n\n"
        f"Anomaly ID: {anomaly_id}\n"
        f"Time: {datetime.now(timezone.utc).isoformat()}\n\n"
        f"{'='*40}\n"
        f"This is an automated alert from the Crypto Prediction System."
    )
    
    sns.publish(
        TopicArn=SNS_TOPIC_ARN,
        Subject=subject[:100],  # SNS subject max 100 chars
        Message=message,
        MessageAttributes={
            'confidence': {
                'DataType': 'Number',
                'StringValue': str(confidence)
            },
            'prediction': {
                'DataType': 'String',
                'StringValue': prediction_result['prediction']
            }
        }
    )
    
    print(f"Alert sent via SNS: {subject}")


def lambda_handler(event, context):
    """
    Main Lambda entry point.
    Triggered by the Feature Engineering Lambda (async invocation).
    """
    
    try:
        # Parse the incoming event
        features = event.get('features', {})
        candle = event.get('candle', {})
        anomaly_direction = event.get('anomaly_direction', 0)
        
        if not features:
            print("No features provided in event")
            return {"statusCode": 400, "body": "No features provided"}
        
        print(f"Received features for prediction. Z-score: {features.get('return_zscore', 'N/A')}")
        
        # Step 1: Load model (cached after first call)
        model, scaler, config = load_model_from_s3()
        
        # Step 2: Run prediction
        prediction_result = predict(features, model, scaler, config)
        prediction_result['return_zscore'] = features.get('return_zscore', 0)
        
        print(f"Prediction: {prediction_result['prediction']} "
              f"(confidence: {prediction_result['confidence']:.1%})")
        
        # Step 3: Save to DynamoDB
        anomaly_id = save_to_dynamodb(
            prediction_result, features, candle, anomaly_direction
        )
        
        # Step 4: Send alert (only if confidence is high enough)
        send_alert(prediction_result, candle, anomaly_direction, anomaly_id)
        
        return {
            "statusCode": 200,
            "body": json.dumps({
                "anomaly_id": anomaly_id,
                "prediction": prediction_result['prediction'],
                "confidence": prediction_result['confidence']
            })
        }
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
