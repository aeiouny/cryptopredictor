#!/usr/bin/env python3
"""
Crypto Breakout vs Reversal ML Pipeline
Fetches historical data, engineers features, detects anomalies,
labels outcomes, and trains 3 supervised models.

Setup: pip install pandas numpy scikit-learn requests joblib matplotlib
"""

import pandas as pd
import numpy as np
import requests
import time
import json
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import joblib


# ===================== STEP 1: DATA COLLECTION =====================
# Pull hourly OHLCV candles from Binance (free, no API key needed)

def fetch_binance_historical(symbol="BTCUSDT", interval="1h", days_back=365):
    # Use binance.us if you're in the US
    base_url = "https://api.binance.us"
    endpoint = f"{base_url}/api/v3/klines"
    
    all_data = []
    end_time = int(datetime.utcnow().timestamp() * 1000)
    start_time = int((datetime.utcnow() - timedelta(days=days_back)).timestamp() * 1000)
    current_start = start_time
    
    print(f"Fetching {symbol} {interval} candles for the last {days_back} days...")
    
    while current_start < end_time:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_time,
            "limit": 1000
        }
        
        try:
            response = requests.get(endpoint, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                break
                
            all_data.extend(data)
            last_close_time = data[-1][6]
            current_start = last_close_time + 1
            
            print(f"  Fetched {len(all_data)} candles so far...")
            
            if len(data) < 1000:
                break
                
        except requests.exceptions.RequestException as e:
            print(f"  Error: {e}")
            break
        
        time.sleep(0.5)
    
    if not all_data:
        print("ERROR: No data fetched.")
        return pd.DataFrame()
    
    # Parse the raw Binance response into a clean dataframe
    df = pd.DataFrame(all_data, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'num_trades', 
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    
    df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)
    
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
    df = df.sort_values('timestamp').reset_index(drop=True)
    df = df.drop_duplicates(subset='timestamp').reset_index(drop=True)
    
    print(f"\nTotal records: {len(df)}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    return df


# ===================== STEP 2: FEATURE ENGINEERING =====================
# Turn raw prices into features the model can learn from

def compute_features(df, window_short=7, window_long=30):
    df = df.copy()
    
    # Log returns - measures price change between candles
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    # Volatility - how jumpy the price has been recently
    df['volatility_short'] = df['log_return'].rolling(window=window_short).std()
    df['volatility_long'] = df['log_return'].rolling(window=window_long).std()
    df['volatility_ratio'] = df['volatility_short'] / df['volatility_long']
    
    # Volume spike - is trading volume unusually high right now
    df['volume_ma_short'] = df['volume'].rolling(window=window_short).mean()
    df['volume_ma_long'] = df['volume'].rolling(window=window_long).mean()
    df['volume_spike'] = df['volume'] / df['volume_ma_long']
    
    # Moving averages - smoothed price trends
    df['ma_short'] = df['close'].rolling(window=window_short).mean()
    df['ma_long'] = df['close'].rolling(window=window_long).mean()
    df['ma_ratio'] = df['ma_short'] / df['ma_long']
    
    # Momentum - is the price trending up or down
    df['momentum_short'] = df['close'].pct_change(periods=window_short)
    df['momentum_long'] = df['close'].pct_change(periods=window_long)
    df['roc'] = (df['close'] - df['close'].shift(window_short)) / df['close'].shift(window_short)
    
    # Z-score - how unusual is the current price move (key feature for anomaly detection)
    rolling_mean = df['log_return'].rolling(window=window_long).mean()
    rolling_std = df['log_return'].rolling(window=window_long).std()
    df['return_zscore'] = (df['log_return'] - rolling_mean) / rolling_std
    
    # Price range and close position within the candle
    df['daily_range'] = (df['high'] - df['low']) / df['close']
    df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
    df['close_position'] = df['close_position'].fillna(0.5)
    
    print(f"Computed {len([c for c in df.columns if c not in ['timestamp','open','high','low','close','volume','market_cap']])} features")
    return df


# ===================== STEP 3: ANOMALY DETECTION =====================
# Flag unusual price movements using z-score threshold

def detect_anomalies(df, zscore_threshold=2.0):
    df = df.copy()
    
    # If absolute z-score exceeds threshold, flag as anomaly
    df['is_anomaly'] = (df['return_zscore'].abs() >= zscore_threshold).astype(int)
    df['anomaly_direction'] = np.sign(df['return_zscore'])
    df.loc[df['is_anomaly'] == 0, 'anomaly_direction'] = 0
    
    num_anomalies = df['is_anomaly'].sum()
    print(f"Z-score threshold: {zscore_threshold}")
    print(f"Anomalies detected: {num_anomalies} out of {len(df)} ({num_anomalies/len(df)*100:.1f}%)")
    print(f"  Up spikes: {(df['anomaly_direction'] == 1).sum()}")
    print(f"  Down drops: {(df['anomaly_direction'] == -1).sum()}")
    
    return df


# ===================== STEP 4: LABELING =====================
# Look at what happened after each anomaly to label it breakout or reversal

def label_anomalies(df, breakout_threshold=0.02, lookforward_window=5):
    df = df.copy()
    df['label'] = np.nan
    
    anomaly_indices = df[df['is_anomaly'] == 1].index
    
    for idx in anomaly_indices:
        anomaly_price = df.loc[idx, 'close']
        anomaly_dir = df.loc[idx, 'anomaly_direction']
        
        future_end = min(idx + lookforward_window, len(df) - 1)
        if idx + 1 > future_end:
            continue
            
        future_prices = df.loc[idx + 1 : future_end, 'close']
        if len(future_prices) == 0:
            continue
        
        # Check if price continued in anomaly direction
        if anomaly_dir > 0:
            price_move = (future_prices.max() - anomaly_price) / anomaly_price
        else:
            price_move = (anomaly_price - future_prices.min()) / anomaly_price
        
        # Breakout = price kept going, Reversal = price came back
        if price_move >= breakout_threshold:
            df.loc[idx, 'label'] = 1  # breakout
        else:
            df.loc[idx, 'label'] = 0  # reversal
    
    labeled = df[df['label'].notna()]
    num_breakout = (labeled['label'] == 1).sum()
    num_reversal = (labeled['label'] == 0).sum()
    
    print(f"Breakout threshold: {breakout_threshold*100:.1f}%, Lookforward: {lookforward_window} periods")
    print(f"Labeled: {len(labeled)} (Breakouts: {num_breakout}, Reversals: {num_reversal})")
    
    if num_breakout == 0 or num_reversal == 0:
        print("WARNING: Only one class present. Adjust thresholds.")
    
    return df


# ===================== STEP 5: MODEL TRAINING =====================
# Train 3 supervised models and compare them

def train_model(df):
    model_data = df[df['label'].notna()].copy()
    
    # These are the features we feed into the models
    feature_columns = [
        'log_return', 'volatility_short', 'volatility_long', 'volatility_ratio',
        'volume_spike', 'ma_ratio', 'momentum_short', 'momentum_long',
        'roc', 'return_zscore', 'daily_range', 'close_position'
    ]
    
    X = model_data[feature_columns].values
    y = model_data['label'].values.astype(int)
    
    print(f"Training data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Classes: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    if len(X) < 30:
        print("WARNING: Small dataset. Results may not be reliable.")
    
    # Scale all features to same range (required for logistic regression)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, 
        stratify=y if len(np.unique(y)) > 1 else None
    )
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # --- Model 1: Logistic Regression ---
    # Draws a straight line to separate breakouts from reversals
    # Simple baseline to compare the other models against
    print("\n--- Logistic Regression (baseline) ---")
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_proba = lr.predict_proba(X_test)[:, 1]
    lr_acc = accuracy_score(y_test, lr_pred)
    print(f"Accuracy: {lr_acc:.3f}")
    
    # --- Model 2: Random Forest ---
    # 100 decision trees that each vote on the outcome, majority wins
    # Good at catching non-linear patterns
    print("\n--- Random Forest ---")
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=5, min_samples_split=5,
        min_samples_leaf=2, random_state=42, class_weight='balanced'
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)[:, 1]
    rf_acc = accuracy_score(y_test, rf_pred)
    print(f"Accuracy: {rf_acc:.3f}")
    
    # --- Model 3: Gradient Boosting ---
    # Builds trees one at a time, each one learns from previous mistakes
    # Usually the most accurate
    print("\n--- Gradient Boosting ---")
    gb = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
    )
    gb.fit(X_train, y_train)
    gb_pred = gb.predict(X_test)
    gb_proba = gb.predict_proba(X_test)[:, 1]
    gb_acc = accuracy_score(y_test, gb_pred)
    print(f"Accuracy: {gb_acc:.3f}")
    
    # Pick the best model by accuracy
    models = {
        'logistic_regression': (lr, lr_acc, lr_pred, lr_proba),
        'random_forest': (rf, rf_acc, rf_pred, rf_proba),
        'gradient_boosting': (gb, gb_acc, gb_pred, gb_proba)
    }
    
    best_name = max(models, key=lambda k: models[k][1])
    best_model, best_acc, best_pred, best_proba = models[best_name]
    
    print(f"\nBEST MODEL: {best_name} ({best_acc:.3f})")
    
    # Detailed results for the best model
    print(classification_report(y_test, best_pred, target_names=['Reversal', 'Breakout'], zero_division=0))
    
    cm = confusion_matrix(y_test, best_pred)
    print(f"Confusion Matrix:")
    print(f"  Predicted:    Reversal  Breakout")
    print(f"  Actual Rev:   {cm[0][0]:>6}    {cm[0][1]:>6}")
    print(f"  Actual Brk:   {cm[1][0]:>6}    {cm[1][1]:>6}")
    
    if len(np.unique(y_test)) > 1:
        auc = roc_auc_score(y_test, best_proba)
        print(f"\nAUC-ROC: {auc:.3f} (1.0 = perfect, 0.5 = random)")
    
    # Show which features matter most
    if best_name in ['random_forest', 'gradient_boosting']:
        importance = best_model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': feature_columns, 'importance': importance
        }).sort_values('importance', ascending=False)
        
        print(f"\nFeature Importance:")
        for _, row in importance_df.iterrows():
            bar = '█' * int(row['importance'] * 50)
            print(f"  {row['feature']:>20}: {row['importance']:.3f} {bar}")
    
    # Cross-validation for more reliable accuracy estimate
    cv_scores = cross_val_score(best_model, X_scaled, y, cv=min(5, len(X)//2), scoring='accuracy')
    print(f"\nCross-Validation: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    
    return {
        'model': best_model,
        'model_name': best_name,
        'scaler': scaler,
        'feature_columns': feature_columns,
        'accuracy': best_acc,
        'cv_scores': cv_scores,
        'all_models': {
            'logistic_regression': {
                'model': lr, 'accuracy': lr_acc, 'predictions': lr_pred,
                'probabilities': lr_proba, 'type': 'supervised',
                'description': 'Draws a straight line to separate breakouts from reversals. Simple baseline.'
            },
            'random_forest': {
                'model': rf, 'accuracy': rf_acc, 'predictions': rf_pred,
                'probabilities': rf_proba, 'type': 'supervised',
                'description': '100 decision trees vote on the outcome. Handles complex patterns.'
            },
            'gradient_boosting': {
                'model': gb, 'accuracy': gb_acc, 'predictions': gb_pred,
                'probabilities': gb_proba, 'type': 'supervised',
                'description': 'Builds trees sequentially, each one fixing previous mistakes.'
            }
        }
    }


# ===================== STEP 6: SAVE MODELS =====================
# Save all 3 models + scaler + config for AWS deployment

def save_model(model_results, output_dir="model_artifacts"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Save best model (used by default in inference)
    joblib.dump(model_results['model'], os.path.join(output_dir, "model.joblib"))
    
    # Save scaler (same one used for all models)
    joblib.dump(model_results['scaler'], os.path.join(output_dir, "scaler.joblib"))
    
    # Save each model individually
    all_models = model_results['all_models']
    model_comparison = []
    
    for name, info in all_models.items():
        joblib.dump(info['model'], os.path.join(output_dir, f"model_{name}.joblib"))
        print(f"Saved {name} ({info['accuracy']:.3f} accuracy)")
        
        model_comparison.append({
            'name': name,
            'accuracy': float(info['accuracy']),
            'type': info['type'],
            'description': info['description'],
            'is_best': (name == model_results['model_name'])
        })
    
    # Save config with model comparison info
    config = {
        'best_model_name': model_results['model_name'],
        'feature_columns': model_results['feature_columns'],
        'best_accuracy': float(model_results['accuracy']),
        'cv_mean_accuracy': float(model_results['cv_scores'].mean()),
        'trained_at': datetime.utcnow().isoformat(),
        'description': 'Crypto breakout vs reversal classifier',
        'all_models': model_comparison
    }
    
    with open(os.path.join(output_dir, "model_config.json"), 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nAll models saved to {output_dir}/")
    print(f"Upload this folder to S3 for the inference Lambda.")


# ===================== STEP 7: TEST INFERENCE =====================
# Simulate what the Lambda will do with new anomaly data

def predict_outcome(features_dict, model, scaler, feature_columns):
    feature_values = [features_dict[col] for col in feature_columns]
    X = np.array(feature_values).reshape(1, -1)
    X_scaled = scaler.transform(X)
    
    prediction = model.predict(X_scaled)[0]
    confidence = model.predict_proba(X_scaled)[0].max()
    
    return {
        'prediction': 'breakout' if prediction == 1 else 'reversal',
        'label': int(prediction),
        'confidence': float(round(confidence, 4)),
    }


# ===================== RUN EVERYTHING =====================
# Run the full pipeline for both BTC and XRP
# Each symbol gets its own model artifacts saved in a subfolder

SYMBOLS = ["BTCUSDT", "XRPUSDT"]

for SYMBOL in SYMBOLS:
    print("\n" + "=" * 60)
    print(f"RUNNING PIPELINE FOR: {SYMBOL}")
    print("=" * 60)

    # Step 1: Data Collection
    print("\n--- STEP 1: DATA COLLECTION ---")
    raw_data = fetch_binance_historical(symbol=SYMBOL, interval="1h", days_back=365)

    if len(raw_data) > 0:
        raw_data.to_csv(f"raw_crypto_data_{SYMBOL}.csv", index=False)
        print(f"Saved {len(raw_data)} rows to raw_crypto_data_{SYMBOL}.csv")
    else:
        print(f"No data for {SYMBOL}, skipping.")
        continue

    # Step 2: Feature Engineering
    print("\n--- STEP 2: FEATURE ENGINEERING ---")
    featured_data = compute_features(raw_data)
    featured_data_clean = featured_data.dropna().reset_index(drop=True)
    print(f"Shape after feature engineering: {featured_data_clean.shape}")

    # Step 3: Anomaly Detection
    print("\n--- STEP 3: ANOMALY DETECTION ---")
    anomaly_data = detect_anomalies(featured_data_clean, zscore_threshold=2.0)

    # Lower threshold if not enough anomalies to train on
    if anomaly_data['is_anomaly'].sum() < 20:
        print("Too few anomalies. Lowering threshold to 1.5...")
        anomaly_data = detect_anomalies(featured_data_clean, zscore_threshold=1.5)
    if anomaly_data['is_anomaly'].sum() < 20:
        print("Still too few. Lowering to 1.0...")
        anomaly_data = detect_anomalies(featured_data_clean, zscore_threshold=1.0)

    # Step 4: Labeling
    print("\n--- STEP 4: LABELING ---")
    labeled_data = label_anomalies(anomaly_data, breakout_threshold=0.02, lookforward_window=5)

    # Step 5: Model Training
    print("\n--- STEP 5: MODEL TRAINING ---")
    model_results = None
    if labeled_data['label'].notna().sum() > 10:
        model_results = train_model(labeled_data)
    else:
        print("Not enough labeled data to train.")
        continue

    # Step 6: Save Models (each symbol gets its own folder)
    print("\n--- STEP 6: SAVE MODELS ---")
    save_model(model_results, output_dir=f"model_artifacts/{SYMBOL}")

    # Step 7: Test Inference
    print("\n--- STEP 7: TEST INFERENCE ---")
    test_anomalies = labeled_data[labeled_data['is_anomaly'] == 1].tail(3)
    print("Testing on recent anomalies:\n")
    for idx, row in test_anomalies.iterrows():
        features = {col: row[col] for col in model_results['feature_columns']}
        result = predict_outcome(features, model_results['model'], model_results['scaler'], model_results['feature_columns'])
        actual = 'breakout' if row.get('label', -1) == 1 else 'reversal'
        print(f"  {row['timestamp']} | Predicted: {result['prediction']} ({result['confidence']:.0%}) | Actual: {actual} | Z-score: {row['return_zscore']:.2f}")

print("\n" + "=" * 60)
print("DONE - Upload model_artifacts/ to S3")
print("  model_artifacts/BTCUSDT/ and model_artifacts/XRPUSDT/")
print("=" * 60)