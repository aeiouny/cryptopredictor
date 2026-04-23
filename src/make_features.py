import pandas as pd
from pathlib import Path

IN_PATH = Path("data/raw/bitcoin.csv")
OUT_PATH = Path("data/processed/bitcoin_features.csv")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(IN_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

df["return"] = df["price"].pct_change()

window = 7
df["rolling_mean"] = df["return"].rolling(window).mean()
df["rolling_std"] = df["return"].rolling(window).std()

df["z_score"] = (df["return"] - df["rolling_mean"]) / df["rolling_std"]

threshold = 2.0
df["is_anomaly"] = df["z_score"].abs() > threshold

df.to_csv(OUT_PATH, index=False)

print(f"Saved processed data to {OUT_PATH}")
print(df[["timestamp", "price", "return", "z_score", "is_anomaly"]].tail(10))
print("Total anomalies:", int(df["is_anomaly"].sum()))