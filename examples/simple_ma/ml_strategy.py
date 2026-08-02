"""Leakage-safe rolling LightGBM signals for the crypto baseline example."""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from qlib.data import D

FEATURES = {
    "return_1d": "$close/Ref($close,1)-1",
    "return_5d": "$close/Ref($close,5)-1",
    "return_10d": "$close/Ref($close,10)-1",
    "return_20d": "$close/Ref($close,20)-1",
    "ma_gap_5d": "$close/Mean($close,5)-1",
    "ma_gap_20d": "$close/Mean($close,20)-1",
    "volatility_10d": "Std($close/Ref($close,1)-1,10)",
    "intraday_return": "$close/$open-1",
    "daily_range": "($high-$low)/$close",
    "volume_ratio_20d": "$volume/Mean($volume,20)-1",
}


def _date_mask(frame, start, end):
    dates = frame.index.get_level_values("datetime")
    return (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))


def build_rolling_lightgbm_signal(
    instruments,
    start,
    end,
    output_dir,
    data_start="2018-01-01",
    horizon=5,
    retrain_days=90,
    validation_days=90,
):
    """Predict five-day open-to-open returns with expanding-window refits.

    A signal dated ``t`` is consumed by Qlib on ``t + 1`` and therefore only
    uses OHLCV fields completed by the close of ``t``.  The label measures the
    return from the next open to the open five days later.  Each refit purges
    ``horizon + 1`` days so every training label is fully known beforehand.
    """
    signal_start = pd.Timestamp(start) - pd.Timedelta(days=1)
    signal_end = pd.Timestamp(end) - pd.Timedelta(days=1)
    label_expr = f"Ref($open,-{horizon + 1})/Ref($open,-1)-1"
    expressions = list(FEATURES.values()) + [label_expr]
    raw = D.features(
        instruments,
        expressions,
        start_time=data_start,
        end_time=end,
    )
    raw = raw.swaplevel().sort_index().replace([np.inf, -np.inf], np.nan)
    raw.columns = list(FEATURES) + ["label"]

    predictions = []
    importance = []
    block_start = signal_start
    while block_start <= signal_end:
        block_end = min(block_start + pd.Timedelta(days=retrain_days - 1), signal_end)
        known_label_end = block_start - pd.Timedelta(days=horizon + 1)
        valid_start = known_label_end - pd.Timedelta(days=validation_days - 1)
        train_end = valid_start - pd.Timedelta(days=1)

        train = raw.loc[_date_mask(raw, data_start, train_end)].dropna(subset=["label"])
        valid = raw.loc[_date_mask(raw, valid_start, known_label_end)].dropna(
            subset=["label"]
        )
        predict = raw.loc[_date_mask(raw, block_start, block_end)]
        if train.empty or valid.empty or predict.empty:
            raise ValueError(
                f"Insufficient data for LightGBM block {block_start.date()}"
            )

        model = lgb.train(
            {
                "objective": "regression",
                "metric": "l2",
                "learning_rate": 0.03,
                "num_leaves": 15,
                "max_depth": 4,
                "min_data_in_leaf": 40,
                "feature_fraction": 1.0,
                "bagging_fraction": 1.0,
                "bagging_freq": 0,
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
            },
            lgb.Dataset(train[list(FEATURES)], label=train["label"]),
            num_boost_round=300,
            valid_sets=[lgb.Dataset(valid[list(FEATURES)], label=valid["label"])],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
        )
        predictions.append(
            pd.Series(
                model.predict(predict[list(FEATURES)]),
                index=predict.index,
                name="score",
            )
        )
        importance.append(
            pd.Series(
                model.feature_importance(importance_type="gain"),
                index=list(FEATURES),
                name=block_start.strftime("%Y-%m-%d"),
            )
        )
        print(
            f"LightGBM {block_start.date()}..{block_end.date()}: "
            f"train through {train_end.date()}, "
            f"best_iteration={model.best_iteration}"
        )
        block_start = block_end + pd.Timedelta(days=1)

    signal = pd.concat(predictions).sort_index()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    signal.rename_axis(index=["datetime", "instrument"]).to_csv(
        output_dir / "lightgbm_predictions.csv"
    )
    pd.concat(importance, axis=1).mean(axis=1).sort_values(ascending=False).rename(
        "mean_gain"
    ).to_csv(output_dir / "lightgbm_feature_importance.csv")
    return signal
