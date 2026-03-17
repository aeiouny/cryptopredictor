# Cloud-Native ML Prediction of Cryptocurrency Breakout vs Reversal

## Problem
Crypto alerts based on static thresholds create too many unnecessary notifications.

## Objective
Build a supervised ML prototype that predicts whether a detected anomaly leads to a breakout or a reversal.

## Inputs
- price returns
- volatility
- moving averages
- momentum
- anomaly score

## Outputs
- breakout or reversal prediction
- confidence score

## MVP
- one cryptocurrency
- historical data first
- z-score anomaly detection
- one baseline classifier
- S3 for raw data
- DynamoDB for prediction results

## Tech Stack
- Python
- Pandas
- NumPy
- scikit-learn
- Coinpaprika API
- AWS Lambda
- Amazon S3
- Amazon DynamoDB
- Amazon EventBridge
- Amazon CloudWatch