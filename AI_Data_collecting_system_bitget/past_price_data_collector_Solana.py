import requests
import time
import json
from datetime import datetime, timedelta

API_URL = "https://api.bitget.com/api/v2/mix/market/history-candles"
SYMBOL = "SOLUSDT"
PRODUCT_TYPE = "USDT-FUTURES"
GRANULARITY = "15m"
LIMIT = 200  # max per request

# 6 months in days (approx)
DAYS = 100
CANDLES_PER_DAY = 96  # 24*60/15
TOTAL_CANDLES = DAYS * CANDLES_PER_DAY

OUTPUT_FILE = "solusdt_1000days.json"

# Helper to convert ms timestamp to ISO8601 with microseconds and 'Z'
def ms_to_iso(ms):
    return datetime.utcfromtimestamp(int(ms) / 1000).strftime("%Y-%m-%dT%H:%M:%S.%f0Z")

def fetch_candles(symbol, product_type, granularity, end_time=None, limit=LIMIT):
    params = {
        "symbol": symbol,
        "productType": product_type,
        "granularity": granularity,
        "limit": str(limit),
    }
    if end_time:
        params["endTime"] = str(end_time)
    resp = requests.get(API_URL, params=params)
    resp.raise_for_status()
    data = resp.json()
    if data.get("code") != "00000":
        raise Exception(f"API error: {data.get('msg')}")
    return data["data"]

def candle_to_dict(candle):
    # candle: [timestamp, open, high, low, close, base_vol, quote_vol]
    return {
        "time_period_start": ms_to_iso(candle[0]),
        "time_period_end": ms_to_iso(str(int(candle[0]) + 15*60*1000)),
        "time_open": ms_to_iso(candle[0]),
        "time_close": ms_to_iso(str(int(candle[0]) + 15*60*1000 - 1)),
        "price_open": float(candle[1]),
        "price_high": float(candle[2]),
        "price_low": float(candle[3]),
        "price_close": float(candle[4]),
        "volume_traded": float(candle[5]),
        "trades_count": 0  # Add trades_count field with default value
    }

def main():
    all_candles = {}
    fetched = 0
    end_time = int(time.time() * 1000)  # now in ms
    print(f"Fetching up to {TOTAL_CANDLES} candles...")
    while fetched < TOTAL_CANDLES:
        candles = fetch_candles(SYMBOL, PRODUCT_TYPE, GRANULARITY, end_time=end_time, limit=LIMIT)
        if not candles:
            print("No more candles returned.")
            break
        for candle in candles:
            ts = int(candle[0])
            if ts not in all_candles:
                all_candles[ts] = candle
        fetched = len(all_candles)
        print(f"Fetched {fetched} unique candles so far...")
        # Prepare for next page: get the earliest timestamp and go back one candle
        min_ts = min(int(c[0]) for c in candles)
        end_time = min_ts - 1
        # Respect API rate limit
        time.sleep(0.06)  # 20 req/sec max
        if len(candles) < LIMIT:
            break  # No more data
    # Sort by timestamp ascending
    sorted_candles = [all_candles[ts] for ts in sorted(all_candles.keys())]
    # Only keep the most recent TOTAL_CANDLES
    sorted_candles = sorted_candles[-TOTAL_CANDLES:]
    # Convert to requested JSON style
    json_data = [candle_to_dict(c) for c in sorted_candles]
    # Save
    with open(OUTPUT_FILE, "w") as f:
        json.dump(json_data, f, indent=4)
    print(f"Saved {len(json_data)} candles to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
