"""Leakage-safe, multi-horizon stacked BTC ensemble.

The short, medium, and long models learn different targets (1, 5, and 20
open-to-open sessions).  Their predictions are combined only after each model
has produced a separate validation prediction; the next test block is never
used to choose an iteration, feature group, or stacker weight.
"""

import argparse
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qlib

from run_bitcoin_signal_ensembles import (
    _date_mask,
    _fit_ridge,
    _predict_ridge,
    build_signal_frame,
    run_one,
    summarize,
)


HORIZONS = {"short_1d": 1, "medium_5d": 5, "long_20d": 20}
FEATURE_GROUPS = {
    "short_1d": [
        "momentum_1d",
        "momentum_5d",
        "momentum_10d",
        "mean_reversion_3d",
        "rsi_7_centered",
        "rsi_14_centered",
        "stochastic_14",
        "bollinger_zscore_20",
        "intraday_return",
        "volume_ratio_20d",
    ],
    "medium_5d": [
        "momentum_5d",
        "momentum_10d",
        "momentum_20d",
        "momentum_50d",
        "ma_gap_20d",
        "ma_gap_50d",
        "ma_cross_5_20",
        "ma_cross_10_50",
        "macd_histogram",
        "rsi_14_centered",
        "rsi_28_centered",
        "breakout_20d",
        "obv_trend_20d",
        "trend_strength_20d",
    ],
    "long_20d": [
        "momentum_20d",
        "momentum_50d",
        "momentum_100d",
        "ma_gap_50d",
        "ma_gap_100d",
        "ma_gap_200d",
        "ma_cross_20_50",
        "macd_histogram",
        "rsi_28_centered",
        "breakout_20d",
        "obv_trend_20d",
        "trend_strength_20d",
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider-uri", default=".data/bitcoin/qlib")
    parser.add_argument("--instrument", default="BTCUSD")
    parser.add_argument("--data-start", default="2017-01-01")
    parser.add_argument("--start", default="2021-01-01", help="OOS start")
    parser.add_argument("--end", default="2025-12-31", help="OOS end")
    parser.add_argument("--retrain-days", type=int, default=90)
    parser.add_argument("--train-days", type=int, default=540)
    parser.add_argument("--early-stop-days", type=int, default=60)
    parser.add_argument("--stack-validation-days", type=int, default=60)
    parser.add_argument("--stack-alpha", type=float, default=10.0)
    parser.add_argument("--open-cost", type=float, default=0.001)
    parser.add_argument("--close-cost", type=float, default=0.001)
    parser.add_argument("--min-holding-days", type=int, default=1)
    parser.add_argument("--annualization-days", type=int, default=365)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("examples/simple_ma/multihorizon_results")
    )
    return parser.parse_args()


def _build_label_frame(instrument, data_start, end):
    """Build one feature frame with three non-overlapping prediction targets."""
    frames = {
        name: build_signal_frame(instrument, data_start, end, horizon)
        for name, horizon in HORIZONS.items()
    }
    features = frames["medium_5d"].drop(columns="label")
    labels = pd.DataFrame(
        {name: frame["label"] for name, frame in frames.items()}, index=features.index
    )
    return pd.concat([features, labels], axis=1)


def _train_lgb(train_x, train_y, early_x, early_y):
    """Select boosting rounds on a period that is separate from stacking data."""
    params = {
        "objective": "regression",
        "metric": "l2",
        "learning_rate": 0.03,
        "num_leaves": 7,
        "max_depth": 3,
        "min_data_in_leaf": 30,
        "feature_fraction": 0.9,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "verbosity": -1,
        "seed": 7,
        "bagging_seed": 7,
        "feature_fraction_seed": 7,
        "data_random_seed": 7,
        "deterministic": True,
        "force_col_wise": True,
        "num_threads": 1,
    }
    model = lgb.train(
        params,
        lgb.Dataset(train_x, label=train_y),
        num_boost_round=1000,
        valid_sets=[lgb.Dataset(early_x, label=early_y)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    best_iteration = model.best_iteration or 1000
    final_model = lgb.train(
        params,
        lgb.Dataset(pd.concat([train_x, early_x]), label=pd.concat([train_y, early_y])),
        num_boost_round=best_iteration,
        callbacks=[lgb.log_evaluation(0)],
    )
    return model, final_model, best_iteration


def build_multihorizon_scores(
    frame, start, end, output_dir, retrain_days, train_days, early_stop_days,
    stack_validation_days, stack_alpha,
):
    """Fit horizon models and a validation-trained stacker in rolling blocks."""
    signal_start = pd.Timestamp(start) - pd.Timedelta(days=1)
    signal_end = pd.Timestamp(end) - pd.Timedelta(days=1)
    longest_horizon = max(HORIZONS.values())
    blocks, diagnostics, weights = [], [], []
    block_start = signal_start

    while block_start <= signal_end:
        block_end = min(block_start + pd.Timedelta(days=retrain_days - 1), signal_end)
        # A long-20d signal dated t needs the open at t+21, so only labels
        # through this cutoff are known at the start of the test block.
        known_label_end = block_start - pd.Timedelta(days=longest_horizon + 1)
        stack_start = known_label_end - pd.Timedelta(days=stack_validation_days - 1)
        early_end = stack_start - pd.Timedelta(days=1)
        early_start = early_end - pd.Timedelta(days=early_stop_days - 1)
        train_end = early_start - pd.Timedelta(days=1)
        train_start = train_end - pd.Timedelta(days=train_days - 1)

        train = frame.loc[_date_mask(frame, train_start, train_end)].dropna()
        early = frame.loc[_date_mask(frame, early_start, early_end)].dropna()
        stack = frame.loc[_date_mask(frame, stack_start, known_label_end)].dropna()
        predict = frame.loc[_date_mask(frame, block_start, block_end)].dropna(
            subset=sum(FEATURE_GROUPS.values(), [])
        )
        if min(len(train), len(early), len(stack)) < 60 or predict.empty:
            raise ValueError(
                f"Insufficient history for block {block_start.date()}. "
                "Use an earlier data-start or a later OOS start."
            )

        stack_predictions, test_predictions, best_iterations = {}, {}, {}
        for name, horizon in HORIZONS.items():
            columns = FEATURE_GROUPS[name]
            selected, final_model, best_iteration = _train_lgb(
                train[columns], train[name], early[columns], early[name]
            )
            # These are out-of-sample with respect to the base model's train
            # and early-stop segments, so they are safe data for the stacker.
            stack_predictions[name] = selected.predict(stack[columns])
            test_predictions[name] = final_model.predict(predict[columns])
            best_iterations[name] = best_iteration

        stack_x = pd.DataFrame(stack_predictions, index=stack.index)
        test_x = pd.DataFrame(test_predictions, index=predict.index)
        stack_mean = stack_x.mean(axis=0)
        stack_scale = stack_x.std(axis=0).replace(0, 1).fillna(1)
        stack_x_z = (stack_x - stack_mean) / stack_scale
        test_x_z = (test_x - stack_mean) / stack_scale
        ridge_mean, ridge_scale, ridge_coef = _fit_ridge(
            stack_x_z, stack["medium_5d"], stack_alpha
        )
        stacked = _predict_ridge(test_x_z, ridge_mean, ridge_scale, ridge_coef)
        # A second transparent output lets us distinguish learned stacking from
        # simple multi-horizon agreement.  Long horizon is a regime gate.
        equal_weight = test_x_z.mean(axis=1)
        long_positive = test_x["long_20d"] > 0
        regime_gated = np.where(long_positive, stacked, -np.abs(stacked))

        blocks.append(
            pd.DataFrame(
                {
                    "multihorizon_equal": equal_weight,
                    "multihorizon_stacked": stacked,
                    "multihorizon_regime_gated": regime_gated,
                },
                index=predict.index,
            )
        )
        weights.append(
            pd.Series(
                ridge_coef,
                index=["intercept"] + list(HORIZONS),
                name=block_start.strftime("%Y-%m-%d"),
            )
        )
        diagnostics.append(
            {
                "block_start": block_start,
                "block_end": block_end,
                "train_rows": len(train),
                "early_stop_rows": len(early),
                "stack_validation_rows": len(stack),
                **{f"{name}_best_iteration": value for name, value in best_iterations.items()},
            }
        )
        print(
            f"Refit {block_start.date()}..{block_end.date()}: train={len(train)}, "
            f"early={len(early)}, stack={len(stack)}, iterations={best_iterations}"
        )
        block_start = block_end + pd.Timedelta(days=1)

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(diagnostics).to_csv(output_dir / "multihorizon_refit_log.csv", index=False)
    pd.concat(weights, axis=1).to_csv(output_dir / "multihorizon_stacker_weights.csv")
    return pd.concat(blocks).sort_index()


def main():
    args = parse_args()
    if min(args.retrain_days, args.train_days, args.early_stop_days, args.stack_validation_days) < 1:
        raise ValueError("all rolling-window lengths must be positive")
    qlib.init(provider_uri=args.provider_uri, region="us", kernels=1)
    frame = _build_label_frame(args.instrument, args.data_start, args.end)
    scores = build_multihorizon_scores(
        frame,
        args.start,
        args.end,
        args.output_dir,
        args.retrain_days,
        args.train_days,
        args.early_stop_days,
        args.stack_validation_days,
        args.stack_alpha,
    )
    scores.to_csv(args.output_dir / "multihorizon_scores.csv")

    reports, growths, rows = {}, {}, []
    for name in scores:
        print(f"Running {name}...")
        report = run_one(scores[name], args)
        reports[name] = report
        row, growth = summarize(name, report, args.annualization_days)
        rows.append(row)
        growths[name] = growth
    reference = next(iter(reports.values()))
    benchmark_report = reference.copy()
    benchmark_report["return"] = benchmark_report["bench"]
    benchmark_report["cost"] = 0
    benchmark_report["turnover"] = 0
    row, growth = summarize("BTC buy and hold", benchmark_report, args.annualization_days)
    rows.append(row)
    growths["BTC buy and hold"] = growth

    metrics = pd.DataFrame(rows).set_index("strategy").sort_values("sharpe_ratio", ascending=False)
    metrics.to_csv(args.output_dir / "multihorizon_metrics.csv")
    pd.DataFrame(growths).to_csv(args.output_dir / "multihorizon_growth.csv")
    print("\nMulti-horizon OOS results:")
    print(metrics)
    fig, ax = plt.subplots(figsize=(11, 6))
    for name, growth in growths.items():
        style = {"linewidth": 2.2} if name == "multihorizon_stacked" else {}
        if name == "BTC buy and hold":
            style["linestyle"] = "--"
        ax.plot(growth.index, growth, label=name, **style)
    ax.set(title="BTC multi-horizon stacked ensemble", xlabel="Date", ylabel="Growth of $1")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "multihorizon_returns.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
