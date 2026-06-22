"""Download Coinbase daily crypto candles and convert them to Qlib data."""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.dump_bin import DumpDataAll

COINBASE_CANDLES_URL = "https://api.exchange.coinbase.com/products/{product}/candles"
SECONDS_PER_DAY = 86_400
MAX_DAYS_PER_REQUEST = 290  # Coinbase's documented maximum is 300 candles.


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2017-01-01")
    parser.add_argument(
        "--end", default=pd.Timestamp.now(tz="UTC").normalize().date().isoformat()
    )
    parser.add_argument("--output-dir", default=".data/bitcoin")
    parser.add_argument(
        "--products",
        nargs="+",
        default=["BTC-USD"],
        help="Coinbase products to include (for example BTC-USD ETH-USD)",
    )
    return parser.parse_args()


def fetch_candles(start, end, product):
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
            response = session.get(
                COINBASE_CANDLES_URL.format(product=product), params=params, timeout=30
            )
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
        print(f"{product}: downloaded through {chunk_end.date()}")
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
    frame["symbol"] = product.replace("-", "")
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

    expected_days = (end - start).days
    product_frames = []
    for product in args.products:
        candles = fetch_candles(start, end, product)
        if len(candles) < expected_days * 0.99:
            raise RuntimeError(
                f"{product}: only received {len(candles)} of {expected_days} expected daily candles"
            )
        candles["crypto_weight"] = 1.0 / len(args.products)
        candles.to_csv(
            source_dir / f"{product.replace('-', '').lower()}.csv", index=False
        )
        product_frames.append(candles)

    DumpDataAll(
        data_path=str(source_dir),
        qlib_dir=str(qlib_dir),
        freq="day",
        max_workers=1,
        exclude_fields="date,symbol",
    ).dump()

    # Qlib uses the next calendar timestamp to close each execution interval.
    # Keep feature dates unchanged, but provide future 24/7 interval boundaries.
    first_date = min(frame["date"].min() for frame in product_frames)
    last_date = max(frame["date"].max() for frame in product_frames)
    future_end = last_date + pd.Timedelta(days=366)
    future_calendar = pd.date_range(first_date, future_end, freq="D")
    future_path = qlib_dir / "calendars" / "day_future.txt"
    pd.Series(future_calendar.strftime("%Y-%m-%d")).to_csv(
        future_path, index=False, header=False
    )

    total_rows = sum(len(frame) for frame in product_frames)
    print(
        f"\nWrote {total_rows} daily candles for {len(product_frames)} products to {qlib_dir}"
    )
    print(pd.concat(product_frames).groupby("symbol").tail(1))


if __name__ == "__main__":
    main()
