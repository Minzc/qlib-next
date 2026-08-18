"""Walk-forward comparison of BTC trading signals and learned ensembles.

Every score dated ``t`` is calculated using the close of ``t`` and is consumed
by the strategy at the next session's open.  A meta-model is refit every
``retrain_days`` sessions using only labels whose complete holding horizon is
known at that point.  This lets the experiment compare signals and learned
combinations on an equal, out-of-sample footing.
"""

import argparse
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qlib
from qlib.backtest import backtest
from qlib.data import D

from strategy import SingleAssetSignalStrategy


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider-uri", default=".data/bitcoin/qlib")
    parser.add_argument("--instrument", default="BTCUSD")
    parser.add_argument("--start", default="2019-01-01", help="OOS start")
    parser.add_argument("--end", default="2025-12-31", help="OOS end")
    parser.add_argument("--data-start", default="2017-01-01")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--retrain-days", type=int, default=90)
    parser.add_argument(
        "--train-days",
        type=int,
        default=730,
        help="Rolling training window; use 0 for expanding-window training.",
    )
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument(
        "--selector-retrain-days",
        type=int,
        default=30,
        help="Maximum sessions between adaptive-expert re-selections.",
    )
    parser.add_argument(
        "--selector-eval-days",
        type=int,
        default=365,
        help="Trailing realized-return window used to score signal experts.",
    )
    parser.add_argument(
        "--selector-monitor-days",
        type=int,
        default=30,
        help="Performance-monitor window that can trigger an early re-selection.",
    )
    parser.add_argument(
        "--selector-min-retrain-days",
        type=int,
        default=14,
        help="Cooldown after any selector refit; prevents daily reaction to noise.",
    )
    parser.add_argument("--min-holding-days", type=int, default=1)
    parser.add_argument("--annualization-days", type=int, default=365)
    parser.add_argument("--open-cost", type=float, default=0.001)
    parser.add_argument("--close-cost", type=float, default=0.001)
    parser.add_argument(
        "--only",
        nargs="+",
        help="Optional strategy names to backtest; BTC buy and hold is always included.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("examples/simple_ma/ensemble_results"))
    return parser.parse_args()


def _by_instrument(frame, func):
    return frame.groupby(level="instrument", group_keys=False).transform(func)


def build_signal_frame(instrument, data_start, end, horizon):
    """Return close-known signal features and a future open-to-open label."""
    raw = D.features(
        [instrument],
        ["$open", "$high", "$low", "$close", "$volume"],
        start_time=data_start,
        end_time=end,
    )
    raw = raw.swaplevel().sort_index()
    raw.columns = ["open", "high", "low", "close", "volume"]
    close, opening, volume = raw["close"], raw["open"], raw["volume"]
    returns = _by_instrument(close, lambda value: value.pct_change())
    ma = lambda window: _by_instrument(
        close, lambda value: value.rolling(window, min_periods=window).mean()
    )
    momentum = lambda window: _by_instrument(
        close, lambda value: value.pct_change(window)
    )
    ema = lambda span: _by_instrument(
        close,
        lambda value: value.ewm(span=span, adjust=False, min_periods=span).mean(),
    )
    rolling_std_20 = _by_instrument(
        returns, lambda value: value.rolling(20, min_periods=20).std()
    )

    def rsi(window):
        gains = returns.clip(lower=0)
        losses = -returns.clip(upper=0)
        mean_gain = _by_instrument(
            gains, lambda value: value.rolling(window, min_periods=window).mean()
        )
        mean_loss = _by_instrument(
            losses, lambda value: value.rolling(window, min_periods=window).mean()
        )
        return 100 * mean_gain / (mean_gain + mean_loss)

    rsi_7, rsi_14, rsi_28 = rsi(7), rsi(14), rsi(28)
    prior_high = _by_instrument(
        close, lambda value: value.shift(1).rolling(20, min_periods=20).max()
    )
    high_14 = _by_instrument(
        raw["high"], lambda value: value.rolling(14, min_periods=14).max()
    )
    low_14 = _by_instrument(
        raw["low"], lambda value: value.rolling(14, min_periods=14).min()
    )
    rolling_volume_20 = _by_instrument(
        volume, lambda value: value.rolling(20, min_periods=20).mean()
    )
    rolling_volume_sum_20 = _by_instrument(
        volume, lambda value: value.rolling(20, min_periods=20).sum()
    )
    obv = (np.sign(_by_instrument(close, lambda value: value.diff())) * volume).groupby(
        level="instrument"
    ).cumsum()
    obv_trend_20 = _by_instrument(obv, lambda value: value.diff(20)) / rolling_volume_sum_20
    macd = ema(12) - ema(26)
    macd_signal = _by_instrument(
        macd,
        lambda value: value.ewm(span=9, adjust=False, min_periods=9).mean(),
    )

    features = pd.DataFrame(
        {
            "momentum_5d": momentum(5),
            "momentum_1d": momentum(1),
            "momentum_10d": momentum(10),
            "momentum_20d": momentum(20),
            "momentum_50d": momentum(50),
            "momentum_100d": momentum(100),
            "mean_reversion_3d": -momentum(3),
            "ma_gap_20d": close / ma(20) - 1,
            "ma_gap_50d": close / ma(50) - 1,
            "ma_gap_100d": close / ma(100) - 1,
            "ma_gap_200d": close / ma(200) - 1,
            "ma_cross_5_20": ma(5) / ma(20) - 1,
            "ma_cross_10_50": ma(10) / ma(50) - 1,
            "ma_cross_20_50": ma(20) / ma(50) - 1,
            "macd_histogram": macd - macd_signal,
            "bollinger_zscore_20": (close - ma(20)) / (2 * rolling_std_20 * close),
            "rsi_7_centered": (rsi_7 - 50) / 50,
            "rsi_14_centered": (rsi_14 - 50) / 50,
            "rsi_28_centered": (rsi_28 - 50) / 50,
            "stochastic_14": (close - low_14) / (high_14 - low_14) - 0.5,
            "breakout_20d": close / prior_high - 1,
            "volume_ratio_20d": volume / rolling_volume_20 - 1,
            "obv_trend_20d": obv_trend_20,
            "trend_strength_20d": momentum(20) / rolling_std_20,
            "intraday_return": close / opening - 1,
        },
        index=raw.index,
    )
    # Directional consensus baselines.  Values are deliberately binary: unlike
    # the regression ensembles, these enter only when independent views agree.
    # They are retained as ordinary candidate signals and are never selected
    # using the final test-period performance.
    features["confirm_ma50_mom20"] = (
        (features["ma_gap_50d"] > 0) & (features["momentum_20d"] > 0)
    ).astype(float) - 0.5
    features["confirm_ma50_rsi"] = (
        (features["ma_gap_50d"] > 0) & (features["rsi_14_centered"] > 0)
    ).astype(float) - 0.5
    features["confirm_ma50_ma100"] = (
        (features["ma_gap_50d"] > 0) & (features["ma_gap_100d"] > 0)
    ).astype(float) - 0.5
    features["vote_ma50_mom20_rsi"] = (
        (
            (features["ma_gap_50d"] > 0).astype(int)
            + (features["momentum_20d"] > 0).astype(int)
            + (features["rsi_14_centered"] > 0).astype(int)
        )
        >= 2
    ).astype(float) - 0.5
    # Signal t is known after close(t); the position runs open(t+1)..open(t+h+1).
    label = _by_instrument(
        opening, lambda value: value.shift(-(horizon + 1)) / value.shift(-1) - 1
    )
    return features.replace([np.inf, -np.inf], np.nan).assign(label=label)


def _one_day_open_return(opening):
    """Return earned by a score at t while held open(t+1)..open(t+2)."""
    return _by_instrument(
        opening, lambda value: value.shift(-2) / value.shift(-1) - 1
    )


def build_adaptive_expert_scores(
    frame,
    start,
    end,
    output_dir,
    evaluation_days,
    retrain_days,
    monitor_days,
    min_retrain_days,
    round_trip_cost,
):
    """Build leakage-safe, periodically reselected signal ensembles.

    Each individual feature is an "expert" that is long when its score is
    positive.  Experts are evaluated on their *realized net open-to-open P&L*,
    including an approximate entry/exit cost.  At most every ``retrain_days``
    sessions, a selector chooses the strongest trailing expert(s).  It also
    refits earlier when the incumbent's monitored return is negative and below
    the cross-sectional median, a practical loss-of-trend/overfit alarm.

    At selection date t, returns through t-2 are the latest fully known
    outcomes, so the two-session purge prevents the selector from seeing the
    return of a score that has not completed its holding period.
    """
    feature_names = frame.columns.drop("label").tolist()
    binary = (frame[feature_names] > 0).astype(float)
    opening = D.features(
        list(frame.index.get_level_values("instrument").unique()),
        ["$open"],
        start_time=frame.index.get_level_values("datetime").min(),
        end_time=frame.index.get_level_values("datetime").max(),
    ).swaplevel().sort_index().iloc[:, 0]
    daily_return = _one_day_open_return(opening).reindex(frame.index)
    net_returns = pd.DataFrame(index=frame.index, columns=feature_names, dtype=float)
    for name in feature_names:
        position = binary[name]
        turnover = (position - position.groupby(level="instrument").shift(1).fillna(0)).abs()
        net_returns[name] = position * daily_return - round_trip_cost * turnover

    score_start = pd.Timestamp(start) - pd.Timedelta(days=1)
    score_end = pd.Timestamp(end) - pd.Timedelta(days=1)
    dates = pd.date_range(score_start, score_end, freq="D")
    generated, decisions = [], []
    active = None
    last_refit = None
    for date in dates:
        if date not in binary.index.get_level_values("datetime"):
            continue
        known_end = date - pd.Timedelta(days=2)
        history_start = known_end - pd.Timedelta(days=evaluation_days - 1)
        history = net_returns.loc[_date_mask(net_returns, history_start, known_end)].dropna(how="all")
        if len(history) < max(90, monitor_days):
            continue
        mean = history.mean()
        std = history.std().replace(0, np.nan)
        sharpe = mean / std * np.sqrt(365)
        ranked = sharpe.sort_values(ascending=False)

        early_refit = False
        if active is not None and len(history) >= monitor_days:
            monitor = history.iloc[-monitor_days:]
            incumbent_return = monitor[active].sum()
            median_return = monitor.sum().median()
            # Both tests are required: avoid switching just because every
            # strategy had a bad period, but react when this expert is weak.
            early_refit = incumbent_return < 0 and incumbent_return < median_return
        elapsed = np.inf if last_refit is None else (date - last_refit).days
        scheduled_refit = elapsed >= retrain_days
        early_refit = early_refit and elapsed >= min_retrain_days
        if scheduled_refit or early_refit:
            active = ranked.index[0]
            top_three = ranked.index[: min(3, len(ranked))]
            positive = sharpe.clip(lower=0)
            if positive.sum() == 0:
                positive[:] = 1.0
            weights = positive / positive.sum()
            last_refit = date
            decisions.append(
                {
                    "date": date,
                    "reason": "early_loss_of_trend" if early_refit and not scheduled_refit else "scheduled",
                    "selected_expert": active,
                    "selected_sharpe": sharpe[active],
                    "monitor_return": incumbent_return if early_refit else np.nan,
                    "median_monitor_return": median_return if early_refit else np.nan,
                    "top_three": ",".join(top_three),
                }
            )
        today = binary.loc[_date_mask(binary, date, date)]
        # A selected expert retains its native score.  The two voting variants
        # only enter when a majority / weighted majority agrees on direction.
        generated.append(
            pd.DataFrame(
                {
                    "adaptive_best_expert": frame.loc[today.index, active],
                    "adaptive_top3_vote": today[top_three].mean(axis=1) - 0.5,
                    "adaptive_sharpe_vote": today.mul(weights, axis=1).sum(axis=1) - 0.5,
                },
                index=today.index,
            )
        )
    if not generated:
        raise ValueError("No adaptive expert scores were generated; increase training history.")
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(decisions).to_csv(output_dir / "adaptive_retrain_log.csv", index=False)
    return pd.concat(generated).sort_index()


def _date_mask(frame, start, end):
    dates = frame.index.get_level_values("datetime")
    return (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))


def _fit_ridge(train_x, train_y, alpha):
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0).replace(0, 1).fillna(1)
    x = ((train_x - mean) / scale).to_numpy()
    design = np.column_stack([np.ones(len(x)), x])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0  # Do not shrink the intercept.
    coef = np.linalg.solve(design.T @ design + penalty, design.T @ train_y.to_numpy())
    return mean, scale, coef


def _predict_ridge(frame, mean, scale, coef):
    x = ((frame - mean) / scale).to_numpy()
    return np.column_stack([np.ones(len(x)), x]) @ coef


def build_walk_forward_ensembles(frame, start, end, output_dir, horizon, retrain_days, train_days, ridge_alpha):
    """Generate equal-weight, Ridge, and nonlinear LightGBM OOS scores."""
    feature_names = frame.columns.drop("label").tolist()
    signal_start = pd.Timestamp(start) - pd.Timedelta(days=1)
    signal_end = pd.Timestamp(end) - pd.Timedelta(days=1)
    predictions, weights = [], []
    block_start = signal_start

    while block_start <= signal_end:
        block_end = min(block_start + pd.Timedelta(days=retrain_days - 1), signal_end)
        known_label_end = block_start - pd.Timedelta(days=horizon + 1)
        train_start = pd.Timestamp(frame.index.get_level_values("datetime").min())
        if train_days:
            train_start = max(train_start, known_label_end - pd.Timedelta(days=train_days - 1))
        train = frame.loc[_date_mask(frame, train_start, known_label_end)].dropna()
        predict = frame.loc[_date_mask(frame, block_start, block_end)].dropna(subset=feature_names)
        if len(train) < max(120, len(feature_names) * 5) or predict.empty:
            raise ValueError(
                f"Insufficient usable data for block {block_start.date()}; "
                "start the test later or provide more training history."
            )

        train_x, train_y = train[feature_names], train["label"]
        mean = train_x.mean(axis=0)
        scale = train_x.std(axis=0).replace(0, 1).fillna(1)
        equal_weight = ((predict[feature_names] - mean) / scale).mean(axis=1)

        ridge_mean, ridge_scale, ridge_coef = _fit_ridge(train_x, train_y, ridge_alpha)
        ridge = _predict_ridge(predict[feature_names], ridge_mean, ridge_scale, ridge_coef)
        weights.append(
            pd.Series(
                ridge_coef,
                index=["intercept"] + feature_names,
                name=block_start.strftime("%Y-%m-%d"),
            )
        )

        model = lgb.train(
            {
                "objective": "regression",
                "metric": "l2",
                "learning_rate": 0.03,
                "num_leaves": 7,
                "max_depth": 3,
                "min_data_in_leaf": 40,
                "lambda_l1": 0.1,
                "lambda_l2": 1.0,
                "verbosity": -1,
                "seed": 7,
                "deterministic": True,
                "force_col_wise": True,
                "num_threads": 1,
            },
            lgb.Dataset(train_x, label=train_y),
            num_boost_round=100,
        )
        predictions.append(
            pd.DataFrame(
                {
                    "equal_weight_zscore": equal_weight,
                    "ridge": ridge,
                    "lightgbm_meta": model.predict(predict[feature_names]),
                },
                index=predict.index,
            )
        )
        print(
            f"Refit {block_start.date()}..{block_end.date()}: "
            f"train {train_start.date()}..{known_label_end.date()} ({len(train)} rows)"
        )
        block_start = block_end + pd.Timedelta(days=1)

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(weights, axis=1).to_csv(output_dir / "ridge_weights.csv")
    return pd.concat(predictions).sort_index()


def run_one(signal, args):
    strategy = SingleAssetSignalStrategy(
        signal=signal,
        min_holding_days=args.min_holding_days,
        min_score=0,
    )
    portfolio, _ = backtest(
        start_time=args.start,
        end_time=args.end,
        strategy=strategy,
        executor={
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
        },
        benchmark=args.instrument,
        account=1_000_000,
        exchange_kwargs={
            "freq": "day",
            "codes": [args.instrument],
            "deal_price": "open",
            "open_cost": args.open_cost,
            "close_cost": args.close_cost,
            "min_cost": 0,
            "trade_unit": None,
            "limit_threshold": None,
        },
    )
    return portfolio["1day"][0]


def summarize(name, report, annualization_days):
    daily = report["return"] - report["cost"]
    growth = (1 + daily).cumprod()
    drawdown = growth / growth.cummax() - 1
    volatility = daily.std()
    return {
        "strategy": name,
        "total_return": growth.iloc[-1] - 1,
        "annualized_return": growth.iloc[-1] ** (annualization_days / len(growth)) - 1,
        "annualized_volatility": volatility * np.sqrt(annualization_days),
        "sharpe_ratio": daily.mean() / volatility * np.sqrt(annualization_days),
        "max_drawdown": drawdown.min(),
        "mean_daily_turnover": report.get("turnover", pd.Series(0, index=report.index)).mean(),
        "final_value_of_1": growth.iloc[-1],
    }, growth


def main():
    args = parse_args()
    if (
        args.horizon < 1
        or args.retrain_days < 1
        or args.train_days < 0
        or args.selector_retrain_days < 1
        or args.selector_eval_days < 1
        or args.selector_monitor_days < 1
        or args.selector_min_retrain_days < 1
    ):
        raise ValueError("all retrain/evaluation windows must be positive and train-days cannot be negative")
    qlib.init(provider_uri=args.provider_uri, region="us", kernels=1)
    frame = build_signal_frame(args.instrument, args.data_start, args.end, args.horizon)
    ensemble = build_walk_forward_ensembles(
        frame, args.start, args.end, args.output_dir, args.horizon,
        args.retrain_days, args.train_days, args.ridge_alpha,
    )
    adaptive = build_adaptive_expert_scores(
        frame,
        args.start,
        args.end,
        args.output_dir,
        args.selector_eval_days,
        args.selector_retrain_days,
        args.selector_monitor_days,
        args.selector_min_retrain_days,
        args.open_cost + args.close_cost,
    )

    # All baselines and ensembles use the same signal date, execution, costs, and rule:
    # long BTC when score > 0; otherwise hold cash.
    scores = pd.concat([frame.drop(columns="label"), ensemble, adaptive], axis=1)
    scores = scores.loc[_date_mask(scores, pd.Timestamp(args.start) - pd.Timedelta(days=1), pd.Timestamp(args.end) - pd.Timedelta(days=1))]
    scores.rename_axis(index=["datetime", "instrument"]).to_csv(args.output_dir / "scores.csv")

    reports, growths, summary = {}, {}, []
    if args.only:
        unknown = sorted(set(args.only) - set(scores.columns))
        if unknown:
            raise ValueError(f"Unknown strategy names: {', '.join(unknown)}")
        scores = scores[args.only]
    for name in scores.columns:
        print(f"Running {name}...")
        report = run_one(scores[name].dropna(), args)
        reports[name] = report
        row, growth = summarize(name, report, args.annualization_days)
        summary.append(row)
        growths[name] = growth

    reference = next(iter(reports.values()))
    benchmark_report = reference.copy()
    benchmark_report["return"] = benchmark_report["bench"]
    benchmark_report["cost"] = 0
    benchmark_report["turnover"] = 0
    row, benchmark_growth = summarize("BTC buy and hold", benchmark_report, args.annualization_days)
    summary.append(row)
    growths["BTC buy and hold"] = benchmark_growth

    metrics = pd.DataFrame(summary).set_index("strategy").sort_values("sharpe_ratio", ascending=False)
    metrics.to_csv(args.output_dir / "metrics.csv")
    pd.DataFrame(growths).to_csv(args.output_dir / "growth.csv")
    print("\nOut-of-sample results:")
    print(metrics)

    fig, ax = plt.subplots(figsize=(12, 6.5))
    for name, growth in growths.items():
        style = {"linewidth": 2.4} if name in {"ridge", "lightgbm_meta", "BTC buy and hold"} else {"alpha": 0.65}
        if name == "BTC buy and hold":
            style["linestyle"] = "--"
        ax.plot(growth.index, growth, label=name, **style)
    ax.set(title="BTC signal and walk-forward ensemble comparison", xlabel="Date", ylabel="Growth of $1")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output_dir / "returns.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
