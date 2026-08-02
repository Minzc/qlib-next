"""Download Coinbase BTC-USD hourly candles for execution-strategy tests."""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.dump_bin import DumpDataAll

URL = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
GRANULARITY = 3_600
MAX_HOURS_PER_REQUEST = 290


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2025-03-01")
    parser.add_argument("--output-dir", default=".data/bitcoin_hourly")
    return parser.parse_args()


def fetch_candles(start, end):
    session = requests.Session()
    session.headers.update({"User-Agent": "qlib-strategy-baselines/1.0"})
    rows = []
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + pd.Timedelta(hours=MAX_HOURS_PER_REQUEST), end)
        params = {
            "granularity": GRANULARITY,
            "start": cursor.isoformat().replace("+00:00", "Z"),
            "end": chunk_end.isoformat().replace("+00:00", "Z"),
        }
        for attempt in range(5):
            response = session.get(URL, params=params, timeout=30)
            if response.status_code == 200:
                break
            if response.status_code == 429 or response.status_code >= 500:
                time.sleep(2**attempt)
                continue
            response.raise_for_status()
        else:
            response.raise_for_status()
        rows.extend(response.json())
        cursor = chunk_end
        time.sleep(0.15)

    frame = pd.DataFrame(
        rows, columns=["time", "low", "high", "open", "close", "volume"]
    )
    frame["date"] = pd.to_datetime(
        frame.pop("time"), unit="s", utc=True
    ).dt.tz_localize(None)
    frame = frame[
        (frame["date"] >= start.tz_localize(None))
        & (frame["date"] < end.tz_localize(None))
    ]
    frame = frame.drop_duplicates("date").sort_values("date").reset_index(drop=True)
    frame["symbol"] = "BTCUSD"
    frame["factor"] = 1.0
    frame["change"] = frame["close"].pct_change()
    return frame[
        ["date", "symbol", "open", "high", "low", "close", "volume", "factor", "change"]
    ]


def main():
    args = parse_args()
    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")
    if start >= end:
        raise ValueError("start must be before end")

    output_dir = Path(args.output_dir)
    source_dir = output_dir / "source"
    qlib_dir = output_dir / "qlib"
    source_dir.mkdir(parents=True, exist_ok=True)

    candles = fetch_candles(start, end)
    expected_hours = int((end - start) / pd.Timedelta(hours=1))
    if len(candles) < expected_hours * 0.99:
        raise RuntimeError(
            f"Only received {len(candles)} of {expected_hours} expected hourly candles"
        )
    candles.to_csv(source_dir / "btcusd.csv", index=False)
    DumpDataAll(
        data_path=str(source_dir),
        qlib_dir=str(qlib_dir),
        freq="60min",
        max_workers=1,
        exclude_fields="date,symbol",
    ).dump()
    future_end = candles["date"].max() + pd.Timedelta(days=30)
    future_calendar = pd.date_range(candles["date"].min(), future_end, freq="h")
    pd.Series(future_calendar.strftime("%Y-%m-%d %H:%M:%S")).to_csv(
        qlib_dir / "calendars" / "60min_future.txt", index=False, header=False
    )
    print(f"Wrote {len(candles)} BTC-USD hourly candles to {qlib_dir}")


if __name__ == "__main__":
    main()
