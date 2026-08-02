"""Compare Qlib order-execution strategies on BTC-USD hourly candles."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import qlib
from qlib.backtest import backtest
from qlib.contrib.strategy.rule_strategy import FileOrderStrategy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--daily-provider", default=".data/crypto_baselines/qlib")
    parser.add_argument("--hourly-provider", default=".data/bitcoin_hourly/qlib")
    parser.add_argument("--start", default="2025-01-15")
    parser.add_argument("--end", default="2025-02-15")
    parser.add_argument("--order-amount", type=float, default=0.05)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("examples/simple_ma/baseline_results")
    )
    return parser.parse_args()


def make_orders(start, end, amount):
    # The final date is kept as an execution boundary, not a new order date.
    dates = pd.date_range(start, pd.Timestamp(end) - pd.Timedelta(days=1), freq="D")
    return pd.DataFrame(
        {
            "datetime": dates,
            "instrument": "BTCUSD",
            "amount": amount,
            "direction": "buy",
        }
    )


def run_one(name, inner_strategy, orders, start, end):
    outer_strategy = FileOrderStrategy(file=orders)
    executor = {
        "class": "NestedExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {
            "time_per_step": "day",
            "inner_executor": {
                "class": "SimulatorExecutor",
                "module_path": "qlib.backtest.executor",
                "kwargs": {
                    "time_per_step": "60min",
                    "generate_portfolio_metrics": True,
                    "indicator_config": {"show_indicator": False},
                },
            },
            "inner_strategy": inner_strategy,
            "generate_portfolio_metrics": True,
            "indicator_config": {"show_indicator": False},
        },
    }
    print(f"\nRunning {name}...")
    portfolio, indicators = backtest(
        start_time=start,
        end_time=end,
        strategy=outer_strategy,
        executor=executor,
        benchmark="BTCUSD",
        account=1_000_000,
        exchange_kwargs={
            "freq": "60min",
            "codes": ["BTCUSD"],
            "deal_price": "close",
            "open_cost": 0,
            "close_cost": 0,
            "min_cost": 0,
            "trade_unit": None,
            "limit_threshold": None,
        },
    )
    indicator, summary = indicators["1day"]
    result = indicator[["ffr", "pa"]].dropna().copy()
    result["pa"] *= 10_000
    result["strategy"] = name
    print(summary)
    return result


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    qlib.init(
        provider_uri={"day": args.daily_provider, "60min": args.hourly_provider},
        region="us",
        kernels=1,
    )
    orders = make_orders(args.start, args.end, args.order_amount)
    strategies = {
        "TWAP": {
            "class": "TWAPStrategy",
            "module_path": "qlib.contrib.strategy.rule_strategy",
        },
        "SBB-EMA": {
            "class": "SBBStrategyEMA",
            "module_path": "qlib.contrib.strategy.rule_strategy",
            "kwargs": {"instruments": ["BTCUSD"], "freq": "60min"},
        },
    }
    results = [
        run_one(name, strategy, orders, args.start, args.end)
        for name, strategy in strategies.items()
    ]
    result = pd.concat(results).rename_axis("date").reset_index()
    result.to_csv(args.output_dir / "execution_metrics.csv", index=False)

    summary = result.groupby("strategy")[["ffr", "pa"]].mean()
    summary.to_csv(args.output_dir / "execution_summary.csv")
    print("\nExecution baseline results:")
    print(summary)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for name, frame in result.groupby("strategy"):
        axes[0].plot(frame["date"], frame["ffr"], marker="o", markersize=3, label=name)
        axes[1].plot(frame["date"], frame["pa"], marker="o", markersize=3, label=name)
    axes[0].set(title="BTC order completion", ylabel="Fill rate (FFR)")
    axes[1].set(
        title="Execution price advantage",
        xlabel="Order date",
        ylabel="Price advantage (bps)",
    )
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "execution_comparison.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
