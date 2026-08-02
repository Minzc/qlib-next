"""Run the minimal moving-average crossover backtest."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import qlib
from qlib.backtest import backtest
from qlib.contrib.evaluate import risk_analysis

from strategy import MovingAverageCrossStrategy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider-uri", default="~/.qlib/qlib_data/cn_data")
    parser.add_argument("--instrument", default="SH600000")
    parser.add_argument("--benchmark", default="SH000300")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2020-12-31")
    parser.add_argument("--fast", type=int, default=5)
    parser.add_argument("--slow", type=int, default=20)
    parser.add_argument("--region", choices=["cn", "us"], default="cn")
    parser.add_argument("--account", type=float, default=1_000_000)
    parser.add_argument("--deal-price", choices=["open", "close"], default="close")
    parser.add_argument("--open-cost", type=float, default=0.0005)
    parser.add_argument("--close-cost", type=float, default=0.0015)
    parser.add_argument("--min-cost", type=float, default=5)
    parser.add_argument(
        "--figure",
        type=Path,
        help="Save a cumulative-return chart (strategy vs. buy-and-hold)",
    )
    parser.add_argument(
        "--trade-unit",
        type=float,
        default=100,
        help="Minimum quantity; use 0 for fractional trading",
    )
    return parser.parse_args()


def save_return_figure(report, output_path, instrument):
    """Plot growth of $1 for the strategy and the buy-and-hold benchmark."""
    strategy_growth = (1 + report["return"] - report["cost"]).cumprod()
    buy_hold_growth = (1 + report["bench"]).cumprod()
    strategy_growth /= strategy_growth.iloc[0]
    buy_hold_growth /= buy_hold_growth.iloc[0]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(
        strategy_growth.index,
        strategy_growth,
        label="MA strategy (after fees)",
        linewidth=2,
    )
    ax.plot(
        buy_hold_growth.index,
        buy_hold_growth,
        label=f"{instrument} buy and hold",
        linewidth=2,
    )
    ax.set(title="Growth of $1", xlabel="Date", ylabel="Portfolio value")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"\nReturn figure saved to {output_path}")


def main():
    args = parse_args()
    qlib.init(provider_uri=args.provider_uri, region=args.region)

    strategy = MovingAverageCrossStrategy(
        instrument=args.instrument,
        fast_window=args.fast,
        slow_window=args.slow,
    )
    executor = {
        "class": "SimulatorExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
    }
    portfolio, _ = backtest(
        start_time=args.start,
        end_time=args.end,
        strategy=strategy,
        executor=executor,
        benchmark=args.benchmark,
        account=args.account,
        exchange_kwargs={
            "freq": "day",
            "codes": [args.instrument],
            "deal_price": args.deal_price,
            "open_cost": args.open_cost,
            "close_cost": args.close_cost,
            "min_cost": args.min_cost,
            "trade_unit": args.trade_unit or None,
        },
    )

    report, _ = portfolio["1day"]
    net_return = report["return"] - report["cost"]
    print("\nLast five portfolio rows:")
    print(report.tail())
    print("\nStrategy risk metrics (after costs):")
    print(risk_analysis(net_return, freq="day"))
    if args.figure:
        save_return_figure(report, args.figure, args.instrument)


if __name__ == "__main__":
    main()
