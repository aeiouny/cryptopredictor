import requests
import pandas as pd
from pathlib import Path
from datetime import datetime

OUT_PATH = Path("data/raw/bitcoin.csv")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

start_ms = int(datetime(2025, 3, 1).timestamp() * 1000)
end_ms = int(datetime(2026, 3, 17).timestamp() * 1000)

url = "https://api.binance.us/api/v3/klines"
params = {
    "symbol": "BTCUSDT",
    "interval": "1d",
    "startTime": start_ms,
    "endTime": end_ms,
    "limit": 1000,
}

response = requests.get(url, params=params, timeout=30)
response.raise_for_status()
data = response.json()

columns = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]
df = pd.DataFrame(data, columns=columns)
df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
for col in ["open", "high", "low", "close", "volume"]:
    df[col] = df[col].astype(float)

df.to_csv(OUT_PATH, index=False)

print(f"Saved {len(df)} rows to {OUT_PATH}")
print(df.head())
