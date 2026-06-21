"""Download Coinbase BTC-USD daily candles and convert them to Qlib data."""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.dump_bin import DumpDataAll

COINBASE_CANDLES_URL = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
SECONDS_PER_DAY = 86_400
MAX_DAYS_PER_REQUEST = 290  # Coinbase's documented maximum is 300 candles.


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2017-01-01")
    parser.add_argument(
        "--end", default=pd.Timestamp.now(tz="UTC").normalize().date().isoformat()
    )
    parser.add_argument("--output-dir", default=".data/bitcoin")
    return parser.parse_args()


def fetch_candles(start, end):
    """Fetch [start, end) daily candles in bounded, retryable requests."""

    session = requests.Session()
    session.headers.update({"User-Agent": "qlib-simple-ma-demo/1.0"})
    chunks = []
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + pd.Timedelta(days=MAX_DAYS_PER_REQUEST), end)
        params = {
            "granularity": SECONDS_PER_DAY,
            "start": cursor.isoformat().replace("+00:00", "Z"),
            "end": chunk_end.isoformat().replace("+00:00", "Z"),
        }
        for attempt in range(5):
            response = session.get(COINBASE_CANDLES_URL, params=params, timeout=30)
            if response.status_code == 200:
                break
            if response.status_code == 429 or response.status_code >= 500:
                time.sleep(2**attempt)
                continue
            response.raise_for_status()
        else:
            response.raise_for_status()

        # Coinbase rows are: time, low, high, open, close, volume.
        chunks.extend(response.json())
        print(f"downloaded through {chunk_end.date()}")
        cursor = chunk_end
        time.sleep(0.15)

    frame = pd.DataFrame(
        chunks, columns=["time", "low", "high", "open", "close", "volume"]
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
    expected_days = (end - start).days
    if len(candles) < expected_days * 0.99:
        raise RuntimeError(
            f"Only received {len(candles)} of {expected_days} expected daily candles"
        )
    csv_path = source_dir / "btcusd.csv"
    candles.to_csv(csv_path, index=False)

    DumpDataAll(
        data_path=str(source_dir),
        qlib_dir=str(qlib_dir),
        freq="day",
        max_workers=1,
        exclude_fields="date,symbol",
    ).dump()

    # Qlib uses the next calendar timestamp to close each execution interval.
    # Keep feature dates unchanged, but provide future 24/7 interval boundaries.
    future_end = candles["date"].max() + pd.Timedelta(days=366)
    future_calendar = pd.date_range(candles["date"].min(), future_end, freq="D")
    future_path = qlib_dir / "calendars" / "day_future.txt"
    pd.Series(future_calendar.strftime("%Y-%m-%d")).to_csv(
        future_path, index=False, header=False
    )

    print(f"\nWrote {len(candles)} BTC-USD candles to {qlib_dir}")
    print(candles.tail())


if __name__ == "__main__":
    main()
