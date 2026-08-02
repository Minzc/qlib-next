"""Compare Qlib portfolio strategies on a reproducible crypto basket."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qlib
from qlib.backtest import backtest
from qlib.contrib.strategy import (
    EnhancedIndexingStrategy,
    SoftTopkStrategy,
    TopkDropoutStrategy,
)
from qlib.data import D
from qlib.model.riskmodel import StructuredCovEstimator

from ml_strategy import build_rolling_lightgbm_signal
from strategy import MovingAverageCrossStrategy, SingleAssetSignalStrategy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider-uri", default=".data/crypto_baselines/qlib")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--data-start", default="2018-01-01")
    parser.add_argument(
        "--instruments", nargs="+", default=["BTCUSD", "ETHUSD", "LTCUSD", "BCHUSD"]
    )
    parser.add_argument("--benchmark", default="BTCUSD")
    parser.add_argument("--benchmark-name", default="BTC buy and hold")
    parser.add_argument("--basket-name", default="Crypto basket buy and hold")
    parser.add_argument("--annualization-days", type=int, default=365)
    parser.add_argument("--title", default="Qlib portfolio strategy baselines")
    parser.add_argument("--ml-cash-threshold", action="store_true")
    parser.add_argument(
        "--riskmodel-root", type=Path, default=Path(".data/crypto_baselines/riskmodel")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("examples/simple_ma/baseline_results")
    )
    return parser.parse_args()


def prepare_risk_model(
    root, instruments, start, end, data_start="2018-01-01", window=60
):
    """Create the rolling statistical risk inputs required by EnhancedIndexingStrategy."""
    prices = (
        D.features(instruments, ["$close"], start_time=data_start, end_time=end)
        .iloc[:, 0]
        .unstack(level="instrument")
    )
    estimator = StructuredCovEstimator(
        num_factors=min(2, len(instruments)), scale_return=False
    )
    for index in range(window, len(prices)):
        date = prices.index[index]
        if date < pd.Timestamp(start) - pd.Timedelta(days=1):
            continue
        returns = prices.iloc[index - window : index + 1].pct_change().iloc[1:]
        returns = returns.replace([np.inf, -np.inf], np.nan).fillna(0)
        factor_exp, factor_cov, specific_var = estimator.predict(
            returns, is_price=False, return_decomposed_components=True
        )
        date_root = root / date.strftime("%Y%m%d")
        date_root.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(factor_exp, index=prices.columns).to_pickle(
            date_root / "factor_exp.pkl"
        )
        pd.DataFrame(np.atleast_2d(factor_cov)).to_pickle(date_root / "factor_cov.pkl")
        pd.Series(
            np.sqrt(np.maximum(specific_var, 1e-12)), index=prices.columns
        ).to_pickle(date_root / "specific_risk.pkl")


def run_one(strategy, instruments, start, end, benchmark):
    executor = {
        "class": "SimulatorExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
    }
    portfolio, _ = backtest(
        start_time=start,
        end_time=end,
        strategy=strategy,
        executor=executor,
        benchmark=benchmark,
        account=1_000_000,
        exchange_kwargs={
            "freq": "day",
            "codes": instruments,
            "deal_price": "open",
            "open_cost": 0.001,
            "close_cost": 0.001,
            "min_cost": 0,
            "trade_unit": None,
            "limit_threshold": None,
        },
    )
    report, _ = portfolio["1day"]
    return report


def growth_from_report(report):
    growth = (1 + report["return"] - report["cost"]).cumprod()
    return growth / growth.iloc[0]


def summarize(name, growth, report, annualization_days):
    daily_return = report["return"] - report["cost"]
    drawdown = growth / growth.cummax() - 1
    volatility = daily_return.std()
    return {
        "strategy": name,
        "total_return": growth.iloc[-1] - 1,
        "annualized_return": growth.iloc[-1] ** (annualization_days / len(growth)) - 1,
        "annualized_volatility": volatility * np.sqrt(annualization_days),
        "sharpe_ratio": daily_return.mean() / volatility * np.sqrt(annualization_days),
        "max_drawdown": drawdown.min(),
        "mean_daily_turnover": report.get(
            "turnover", pd.Series(0, index=report.index)
        ).mean(),
        "final_value_of_1": growth.iloc[-1],
    }


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # EnhancedIndexing performs two tiny feature queries per rebalance. A single
    # worker avoids process-start overhead dominating this four-asset example.
    qlib.init(provider_uri=args.provider_uri, region="us", kernels=1)

    instruments = args.instruments
    signal_start = pd.Timestamp(args.start) - pd.Timedelta(days=40)
    momentum = D.features(
        instruments,
        ["$close/Ref($close, 20)-1"],
        start_time=signal_start,
        end_time=args.end,
    ).iloc[:, 0]
    ml_signal = build_rolling_lightgbm_signal(
        instruments,
        args.start,
        args.end,
        args.output_dir,
        data_start=args.data_start,
    )

    prepare_risk_model(
        args.riskmodel_root, instruments, args.start, args.end, args.data_start
    )
    topk = min(2, len(instruments))
    strategies = {
        f"MA 5/20 ({args.benchmark})": MovingAverageCrossStrategy(
            args.benchmark, fast_window=5, slow_window=20
        ),
        "TopkDropout": TopkDropoutStrategy(
            signal=momentum, topk=topk, n_drop=max(0, topk - 1)
        ),
        "SoftTopk": SoftTopkStrategy(signal=momentum, topk=topk, max_sold_weight=1.0),
        "LightGBM rotation": SingleAssetSignalStrategy(
            signal=ml_signal,
            min_holding_days=5,
            min_score=0 if args.ml_cash_threshold else None,
        ),
    }
    if len(instruments) > 1:
        strategies["EnhancedIndexing"] = EnhancedIndexingStrategy(
            signal=momentum,
            riskmodel_root=str(args.riskmodel_root),
            market="crypto",
            optimizer_kwargs={"lamb": 5, "delta": 0.5, "b_dev": 0.2},
        )
    else:
        print("\nSkipping EnhancedIndexing: it requires a multi-asset universe.")

    reports = {}
    growth = {}
    summary = []
    for name, strategy in strategies.items():
        print(f"\nRunning {name}...")
        report = run_one(strategy, instruments, args.start, args.end, args.benchmark)
        reports[name] = report
        growth[name] = growth_from_report(report)
        summary.append(summarize(name, growth[name], report, args.annualization_days))

    benchmark = (1 + next(iter(reports.values()))["bench"]).cumprod()
    benchmark /= benchmark.iloc[0]
    benchmark_report = next(iter(reports.values())).copy()
    benchmark_report["return"] = benchmark_report["bench"]
    benchmark_report["cost"] = 0
    benchmark_report["turnover"] = 0
    summary.append(
        summarize(
            args.benchmark_name, benchmark, benchmark_report, args.annualization_days
        )
    )

    open_prices = (
        D.features(instruments, ["$open"], start_time=args.start, end_time=args.end)
        .iloc[:, 0]
        .unstack(level="instrument")
    )
    equal_weight = (open_prices / open_prices.iloc[0]).mean(axis=1)
    equal_weight = equal_weight.reindex(benchmark.index).ffill()
    equal_weight /= equal_weight.iloc[0]
    equal_weight_report = benchmark_report.copy()
    equal_weight_report["return"] = equal_weight.pct_change().fillna(0)
    equal_weight_report["turnover"] = 0
    if len(instruments) > 1:
        summary.append(
            summarize(
                args.basket_name,
                equal_weight,
                equal_weight_report,
                args.annualization_days,
            )
        )

    summary_frame = pd.DataFrame(summary).set_index("strategy")
    summary_frame.to_csv(args.output_dir / "portfolio_metrics.csv")
    print("\nPortfolio baseline results:")
    print(summary_frame)

    fig, ax = plt.subplots(figsize=(11, 6))
    for name, series in growth.items():
        ax.plot(series.index, series, label=name, linewidth=1.8)
    ax.plot(
        benchmark.index,
        benchmark,
        label=args.benchmark_name,
        linewidth=2.2,
        linestyle="--",
    )
    growth_output = {**growth, args.benchmark_name: benchmark}
    if len(instruments) > 1:
        ax.plot(
            equal_weight.index,
            equal_weight,
            label=args.basket_name,
            linewidth=2.0,
            linestyle=":",
        )
        growth_output[args.basket_name] = equal_weight
    pd.DataFrame(growth_output).to_csv(args.output_dir / "portfolio_growth.csv")
    ax.set(title=args.title, xlabel="Date", ylabel="Growth of $1")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "portfolio_returns.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
