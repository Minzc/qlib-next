"""Download Yahoo Finance OHLCV candles and convert them to Qlib data."""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.dump_bin import DumpDataAll

YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2026-01-01")
    parser.add_argument("--output-dir", default=".data/qqq")
    parser.add_argument("--symbols", nargs="+", default=["QQQ"])
    parser.add_argument("--interval", choices=["1d", "1h"], default="1d")
    return parser.parse_args()


def fetch_chart(symbol, start, end, interval):
    session = requests.Session()
    session.headers.update({"User-Agent": "qlib-simple-ma-demo/1.0"})
    params = {
        "period1": int(start.timestamp()),
        "period2": int(end.timestamp()),
        "interval": interval,
        "events": "history",
        "includeAdjustedClose": "true",
    }
    for attempt in range(5):
        response = session.get(
            YAHOO_CHART_URL.format(symbol=symbol), params=params, timeout=30
        )
        if response.status_code == 200:
            break
        if response.status_code == 429 or response.status_code >= 500:
            time.sleep(2**attempt)
            continue
        response.raise_for_status()
    else:
        response.raise_for_status()

    result = response.json()["chart"]["result"][0]
    timestamps = result["timestamp"]
    quote = result["indicators"]["quote"][0]
    frame = pd.DataFrame(quote)
    frame["date"] = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(
        "America/New_York"
    )
    if interval == "1d":
        frame["date"] = frame["date"].dt.tz_localize(None).dt.normalize()
    else:
        frame["date"] = frame["date"].dt.tz_localize(None)
    frame["symbol"] = symbol.upper().replace("-", "")
    frame = frame.rename(columns={"open": "open", "high": "high", "low": "low"})
    frame = frame.dropna(subset=["open", "high", "low", "close", "volume"])
    frame = frame[
        (frame["date"] >= start.tz_localize(None))
        & (frame["date"] < end.tz_localize(None))
    ]
    frame = frame.drop_duplicates("date").sort_values("date").reset_index(drop=True)
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

    frames = []
    for symbol in args.symbols:
        candles = fetch_chart(symbol, start, end, args.interval)
        if candles.empty:
            raise RuntimeError(f"{symbol}: no Yahoo candles returned")
        candles.to_csv(
            source_dir / f"{symbol.lower().replace('-', '')}.csv", index=False
        )
        frames.append(candles)
        print(f"{symbol}: downloaded {len(candles)} {args.interval} candles")

    freq = "day" if args.interval == "1d" else "60min"
    DumpDataAll(
        data_path=str(source_dir),
        qlib_dir=str(qlib_dir),
        freq=freq,
        max_workers=1,
        exclude_fields="date,symbol",
    ).dump()

    calendar = pd.Index(pd.concat(frames)["date"].drop_duplicates().sort_values())
    calendar_name = "day_future.txt" if args.interval == "1d" else "60min_future.txt"
    future_path = qlib_dir / "calendars" / calendar_name
    if args.interval == "1d":
        # Use observed exchange trading dates. A plain business-day calendar
        # would include US market holidays and make Qlib request missing opens.
        future = calendar.append(pd.Index([calendar[-1] + pd.offsets.BDay(1)]))
    else:
        future = calendar
    pd.Series(
        future.strftime("%Y-%m-%d" if args.interval == "1d" else "%Y-%m-%d %H:%M:%S")
    ).to_csv(future_path, index=False, header=False)

    total_rows = sum(len(frame) for frame in frames)
    print(
        f"\nWrote {total_rows} {args.interval} candles for {len(frames)} symbols to {qlib_dir}"
    )
    print(pd.concat(frames).groupby("symbol").tail(1))


if __name__ == "__main__":
    main()
