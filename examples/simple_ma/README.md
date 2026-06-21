# Simple moving-average strategy

This beginner example holds one stock when its 5-day simple moving average is
above its 20-day average, and otherwise holds cash. Signals are delayed by one
day to prevent look-ahead bias.

## 1. Install Qlib and download the sample data

From the repository root:

```bash
python -m pip install -e .
python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn
```

## 2. Run the backtest

```bash
python examples/simple_ma/run.py
```

Try other windows or another instrument:

```bash
python examples/simple_ma/run.py --fast 10 --slow 50 --instrument SH600519
```

The example is deliberately small: `strategy.py` contains the trading rule and
`run.py` contains the Qlib initialization, simulator, costs, and result summary.
It is educational code, not investment advice.

## Bitcoin with daily OHLCV candles

Download public BTC-USD candles from Coinbase Exchange and convert them to
Qlib's native binary format:

```bash
python examples/simple_ma/prepare_bitcoin_data.py --start 2017-01-01 --end 2026-01-01
```

Then run the moving-average strategy. It uses yesterday's close-derived signal
and trades at the next daily open, supports fractional BTC, and applies a 0.1%
fee on both buys and sells:

```bash
python examples/simple_ma/run.py \
  --provider-uri .data/bitcoin/qlib \
  --instrument BTCUSD \
  --benchmark BTCUSD \
  --region us \
  --start 2018-01-01 \
  --end 2025-12-31 \
  --deal-price open \
  --trade-unit 0 \
  --open-cost 0.001 \
  --close-cost 0.001 \
  --min-cost 0
```

The raw `volume` field is BTC traded on Coinbase. The dataset is suitable for
testing and education; results from one venue do not represent the entire
Bitcoin market.
