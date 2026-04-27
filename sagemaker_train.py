#!/usr/bin/env python3
"""
SageMaker Training Script
Crypto Breakout vs Reversal ML Pipeline

This script runs inside a SageMaker managed container.
SageMaker passes data via environment variables and standard paths:
  - Input data: /opt/ml/input/data/training/
  - Output model: /opt/ml/model/
  - Hyperparameters: passed as command line args

Usage: Triggered by a SageMaker Training Job, not run directly.
"""

import argparse
import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, mean_absolute_percentage_error


# ===================== ARGUMENT PARSING =====================
# SageMaker passes hyperparameters as command line arguments

def parse_args():
    parser = argparse.ArgumentParser()
    
    # Hyperparameters
    parser.add_argument('--zscore-threshold', type=float, default=2.0)
    parser.add_argument('--breakout-threshold', type=float, default=0.02)
    parser.add_argument('--lookforward-window', type=int, default=5)
    parser.add_argument('--window-short', type=int, default=7)
    parser.add_argument('--window-long', type=int, default=30)
    
    # SageMaker environment paths
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training'))
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output'))
    
    return parser.parse_args()


# ===================== FEATURE ENGINEERING =====================

def compute_features(df, window_short=7, window_long=30):
    df = df.copy()
    
    # Log returns
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    # Volatility
    df['volatility_short'] = df['log_return'].rolling(window=window_short).std()
    df['volatility_long'] = df['log_return'].rolling(window=window_long).std()
    df['volatility_ratio'] = df['volatility_short'] / df['volatility_long']
    
    # Volume spike
    df['volume_ma_long'] = df['volume'].rolling(window=window_long).mean()
    df['volume_spike'] = df['volume'] / df['volume_ma_long']
    
    # Moving averages
    df['ma_short'] = df['close'].rolling(window=window_short).mean()
    df['ma_long'] = df['close'].rolling(window=window_long).mean()
    df['ma_ratio'] = df['ma_short'] / df['ma_long']
    
    # Momentum
    df['momentum_short'] = df['close'].pct_change(periods=window_short)
    df['momentum_long'] = df['close'].pct_change(periods=window_long)
    df['roc'] = (df['close'] - df['close'].shift(window_short)) / df['close'].shift(window_short)
    
    # Z-score
    rolling_mean = df['log_return'].rolling(window=window_long).mean()
    rolling_std = df['log_return'].rolling(window=window_long).std()
    df['return_zscore'] = (df['log_return'] - rolling_mean) / rolling_std
    
    # Price range and close position
    df['daily_range'] = (df['high'] - df['low']) / df['close']
    df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
    df['close_position'] = df['close_position'].fillna(0.5)
    
    return df


# ===================== ANOMALY DETECTION =====================

def detect_anomalies(df, zscore_threshold=2.0):
    df = df.copy()
    df['is_anomaly'] = (df['return_zscore'].abs() >= zscore_threshold).astype(int)
    df['anomaly_direction'] = np.sign(df['return_zscore'])
    df.loc[df['is_anomaly'] == 0, 'anomaly_direction'] = 0
    
    num_anomalies = df['is_anomaly'].sum()
    print(f"Anomalies detected: {num_anomalies} out of {len(df)} ({num_anomalies/len(df)*100:.1f}%)")
    return df


# ===================== LABELING =====================

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
        
        future_prices = df.loc[idx + 1: future_end, 'close']
        if len(future_prices) == 0:
            continue
        
        if anomaly_dir > 0:
            price_move = (future_prices.max() - anomaly_price) / anomaly_price
        else:
            price_move = (anomaly_price - future_prices.min()) / anomaly_price
        
        df.loc[idx, 'label'] = 1 if price_move >= breakout_threshold else 0
    
    labeled = df[df['label'].notna()]
    print(f"Labeled: {len(labeled)} (Breakouts: {(labeled['label']==1).sum()}, Reversals: {(labeled['label']==0).sum()})")
    return df


# ===================== MODEL TRAINING =====================

def train_models(df, symbol):
    model_data = df[df['label'].notna()].copy()
    
    feature_columns = [
        'log_return', 'volatility_short', 'volatility_long', 'volatility_ratio',
        'volume_spike', 'ma_ratio', 'momentum_short', 'momentum_long',
        'roc', 'return_zscore', 'daily_range', 'close_position'
    ]
    
    X = model_data[feature_columns].values
    y = model_data['label'].values.astype(int)
    
    print(f"Training data for {symbol}: {X.shape[0]} samples, {X.shape[1]} features")
    
    if len(X) < 10:
        print(f"Not enough data for {symbol}. Skipping.")
        return None
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42,
        stratify=y if len(np.unique(y)) > 1 else None
    )
    
    # Train all 3 models
    print(f"\n--- Logistic Regression ---")
    lr = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    lr.fit(X_train, y_train)
    lr_acc = accuracy_score(y_test, lr.predict(X_test))
    print(f"Accuracy: {lr_acc:.3f}")
    
    print(f"\n--- Random Forest ---")
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=5,
                                 min_samples_leaf=2, random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test))
    print(f"Accuracy: {rf_acc:.3f}")
    
    print(f"\n--- Gradient Boosting ---")
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    gb.fit(X_train, y_train)
    gb_acc = accuracy_score(y_test, gb.predict(X_test))
    print(f"Accuracy: {gb_acc:.3f}")
    
    # Pick best model
    models = {
        'logistic_regression': (lr, lr_acc),
        'random_forest': (rf, rf_acc),
        'gradient_boosting': (gb, gb_acc)
    }
    best_name = max(models, key=lambda k: models[k][1])
    best_model, best_acc = models[best_name]
    
    print(f"\nBest model for {symbol}: {best_name} ({best_acc:.3f})")
    
    # Cross-validation
    cv_scores = cross_val_score(best_model, X_scaled, y, cv=min(5, len(X)//2), scoring='accuracy')
    print(f"Cross-validation: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    
    # AUC
    if len(np.unique(y_test)) > 1:
        auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
        print(f"AUC-ROC: {auc:.3f}")
    
    return {
        'models': {'logistic_regression': lr, 'random_forest': rf, 'gradient_boosting': gb},
        'best_model': best_model,
        'best_model_name': best_name,
        'scaler': scaler,
        'feature_columns': feature_columns,
        'accuracies': {'logistic_regression': lr_acc, 'random_forest': rf_acc, 'gradient_boosting': gb_acc},
        'best_accuracy': best_acc,
        'cv_mean': float(cv_scores.mean()),
        'price_model': train_price_model(df, scaler, feature_columns)
    }


# ===================== PRICE PREDICTION MODEL =====================

def train_price_model(df, scaler, feature_columns):
    """
    Train a linear regression model to predict next hour's percentage price change.
    Target: how much the price will move (%) in the next candle.
    """
    print(f"\n--- Price Prediction (Linear Regression) ---")
    
    df = df.copy()
    
    # Target: next candle's percentage change
    df['next_pct_change'] = df['close'].pct_change().shift(-1)
    
    # Use all rows with valid features and target (not just anomalies)
    model_data = df[feature_columns + ['next_pct_change']].dropna()
    
    if len(model_data) < 10:
        print("Not enough data for price prediction model.")
        return None
    
    X = model_data[feature_columns].values
    y = model_data['next_pct_change'].values
    
    X_scaled = scaler.transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # Linear Regression for price prediction
    price_model = LinearRegression()
    price_model.fit(X_train, y_train)
    
    y_pred = price_model.predict(X_test)
    
    # MAPE — how far off are our predictions on average (%)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100
    
    # Direction accuracy — did we at least get the direction right?
    direction_acc = np.mean(np.sign(y_pred) == np.sign(y_test))
    
    print(f"Mean Absolute Percentage Error: {mape:.2f}%")
    print(f"Direction accuracy: {direction_acc:.1%} (predicted up/down correctly)")
    
    return {
        'model': price_model,
        'mape': float(mape),
        'direction_accuracy': float(direction_acc)
    }


# ===================== SAVE MODELS =====================

def save_models(results, model_dir, symbol):
    symbol_dir = os.path.join(model_dir, symbol)
    os.makedirs(symbol_dir, exist_ok=True)
    
    # Save best model
    joblib.dump(results['best_model'], os.path.join(symbol_dir, 'model.joblib'))
    
    # Save all classification models individually
    for name, model in results['models'].items():
        joblib.dump(model, os.path.join(symbol_dir, f'model_{name}.joblib'))
    
    # Save scaler
    joblib.dump(results['scaler'], os.path.join(symbol_dir, 'scaler.joblib'))
    
    # Save price prediction model
    price_model_info = results.get('price_model')
    if price_model_info and price_model_info.get('model') is not None:
        joblib.dump(price_model_info['model'], os.path.join(symbol_dir, 'model_price_prediction.joblib'))
        print(f"Saved price prediction model to {symbol_dir}/model_price_prediction.joblib")
    
    # Save config
    config = {
        'best_model_name': results['best_model_name'],
        'feature_columns': results['feature_columns'],
        'best_accuracy': results['best_accuracy'],
        'cv_mean_accuracy': results['cv_mean'],
        'all_models': [
            {'name': k, 'accuracy': v, 'type': 'supervised'}
            for k, v in results['accuracies'].items()
        ],
        'price_prediction': {
            'model': 'linear_regression',
            'type': 'supervised_regression',
            'target': 'next_candle_pct_change',
            'mape': price_model_info['mape'] if price_model_info else None,
            'direction_accuracy': price_model_info['direction_accuracy'] if price_model_info else None
        },
        'trained_at': datetime.utcnow().isoformat(),
        'trained_with': 'SageMaker'
    }
    
    with open(os.path.join(symbol_dir, 'model_config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"All models saved to {symbol_dir}/")


# ===================== MAIN =====================

if __name__ == '__main__':
    args = parse_args()
    
    print("=" * 60)
    print("SAGEMAKER TRAINING JOB STARTED")
    print(f"Model dir: {args.model_dir}")
    print(f"Training data dir: {args.train}")
    print("=" * 60)
    
    symbols = ['BTCUSDT', 'XRPUSDT']
    
    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"TRAINING FOR: {symbol}")
        print(f"{'='*60}")
        
        # Load CSV data from S3 (SageMaker downloads it to the training channel)
        csv_path = os.path.join(args.train, f'raw_crypto_data_{symbol}.csv')
        
        if not os.path.exists(csv_path):
            print(f"Data file not found: {csv_path}")
            print(f"Files in training dir: {os.listdir(args.train)}")
            continue
        
        df = pd.read_csv(csv_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        print(f"Loaded {len(df)} rows for {symbol}")
        
        # Feature engineering
        featured = compute_features(df, args.window_short, args.window_long)
        clean = featured.dropna().reset_index(drop=True)
        
        # Anomaly detection
        threshold = args.zscore_threshold
        anomaly_data = detect_anomalies(clean, threshold)
        
        # Lower threshold if not enough anomalies
        if anomaly_data['is_anomaly'].sum() < 20:
            print("Too few anomalies, lowering threshold to 1.5")
            anomaly_data = detect_anomalies(clean, 1.5)
        if anomaly_data['is_anomaly'].sum() < 20:
            print("Still too few, lowering threshold to 1.0")
            anomaly_data = detect_anomalies(clean, 1.0)
        
        # Labeling
        labeled = label_anomalies(anomaly_data, args.breakout_threshold, args.lookforward_window)
        
        # Train
        if labeled['label'].notna().sum() < 10:
            print(f"Not enough labeled data for {symbol}. Skipping.")
            continue
        
        results = train_models(labeled, symbol)
        
        if results:
            save_models(results, args.model_dir, symbol)
    
    print("\n" + "=" * 60)
    print("SAGEMAKER TRAINING JOB COMPLETE")
    print("Models saved to:", args.model_dir)
    print("=" * 60)