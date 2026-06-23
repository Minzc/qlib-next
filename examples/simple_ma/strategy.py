"""A minimal long/cash moving-average crossover strategy for Qlib."""

import numpy as np
import pandas as pd

from qlib.backtest.decision import Order, TradeDecisionWO
from qlib.data import D
from qlib.strategy.base import BaseStrategy


class MovingAverageCrossStrategy(BaseStrategy):
    """Buy when the fast SMA is above the slow SMA; otherwise hold cash.

    The signal is shifted by one trading bar.  A decision made on day *t*
    therefore only uses prices available at the end of day *t - 1*.
    """

    def __init__(
        self,
        instrument="SH600000",
        fast_window=5,
        slow_window=20,
        risk_degree=0.95,
        **kwargs,
    ):
        if fast_window >= slow_window:
            raise ValueError("fast_window must be smaller than slow_window")
        if not 0 < risk_degree <= 1:
            raise ValueError("risk_degree must be in (0, 1]")

        self.instrument = instrument
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.risk_degree = risk_degree
        self.signal = pd.Series(dtype=float)
        super().__init__(**kwargs)

    def reset_level_infra(self, level_infra):
        super().reset_level_infra(level_infra)
        start_time, _ = self.trade_calendar.get_step_time(0)
        _, end_time = self.trade_calendar.get_step_time(
            self.trade_calendar.get_trade_len() - 1
        )

        # Ref(..., 1) is the important part: today's order never uses today's close.
        expression = (
            f"Ref(Mean($close, {self.fast_window}), 1)"
            f"-Ref(Mean($close, {self.slow_window}), 1)"
        )
        features = D.features(
            [self.instrument],
            [expression],
            start_time=start_time,
            end_time=end_time,
            freq=self.trade_calendar.get_freq(),
        )
        if not features.empty:
            self.signal = features.iloc[:, 0].droplevel("instrument")

    def generate_trade_decision(self, execute_result=None):
        trade_step = self.trade_calendar.get_trade_step()
        start_time, end_time = self.trade_calendar.get_step_time(trade_step)
        signal = self.signal.get(start_time, np.nan)
        if pd.isna(signal):
            return TradeDecisionWO([], self)

        position = self.trade_position
        is_held = self.instrument in position.get_stock_list()
        direction = Order.BUY if signal > 0 else Order.SELL
        if (direction == Order.BUY and is_held) or (
            direction == Order.SELL and not is_held
        ):
            return TradeDecisionWO([], self)
        if not self.trade_exchange.is_stock_tradable(
            self.instrument, start_time, end_time, direction=direction
        ):
            return TradeDecisionWO([], self)

        if direction == Order.SELL:
            amount = position.get_stock_amount(self.instrument)
        else:
            price = self.trade_exchange.get_deal_price(
                self.instrument, start_time, end_time, direction=Order.BUY
            )
            # Leave a cash buffer for commissions and price/rounding effects.
            amount = position.get_cash() * self.risk_degree / price
            factor = self.trade_exchange.get_factor(
                self.instrument, start_time, end_time
            )
            amount = self.trade_exchange.round_amount_by_trade_unit(amount, factor)

        if amount <= 0:
            return TradeDecisionWO([], self)
        order = Order(
            stock_id=self.instrument,
            amount=amount,
            start_time=start_time,
            end_time=end_time,
            direction=direction,
        )
        return TradeDecisionWO([order], self)


class SingleAssetSignalStrategy(BaseStrategy):
    """Hold the highest-scored asset with deterministic five-day turnover."""

    def __init__(self, signal, min_holding_days=5, risk_degree=0.95, **kwargs):
        self.signal = signal.sort_index()
        self.min_holding_days = min_holding_days
        self.risk_degree = risk_degree
        self.buy_step = None
        super().__init__(**kwargs)

    def generate_trade_decision(self, execute_result=None):
        trade_step = self.trade_calendar.get_trade_step()
        start_time, end_time = self.trade_calendar.get_step_time(trade_step)
        signal_time, _ = self.trade_calendar.get_step_time(trade_step, shift=1)
        try:
            scores = self.signal.xs(signal_time, level="datetime").dropna()
        except KeyError:
            return TradeDecisionWO([], self)
        if scores.empty:
            return TradeDecisionWO([], self)

        # Instrument name is an explicit stable tie-breaker.
        target = scores.rename("score").reset_index()
        target = target.sort_values(
            ["score", "instrument"], ascending=[False, True]
        ).iloc[0]["instrument"]
        position = self.trade_position
        held = sorted(position.get_stock_list())

        if held:
            current = held[0]
            if current == target:
                return TradeDecisionWO([], self)
            if (
                self.buy_step is not None
                and trade_step - self.buy_step < self.min_holding_days
            ):
                return TradeDecisionWO([], self)
            if not self.trade_exchange.is_stock_tradable(
                current, start_time, end_time, direction=Order.SELL
            ):
                return TradeDecisionWO([], self)
            amount = position.get_stock_amount(current)
            return TradeDecisionWO(
                [
                    Order(
                        stock_id=current,
                        amount=amount,
                        direction=Order.SELL,
                        start_time=start_time,
                        end_time=end_time,
                    )
                ],
                self,
            )

        if not self.trade_exchange.is_stock_tradable(
            target, start_time, end_time, direction=Order.BUY
        ):
            return TradeDecisionWO([], self)
        price = self.trade_exchange.get_deal_price(
            target, start_time, end_time, direction=Order.BUY
        )
        amount = position.get_cash() * self.risk_degree / price
        factor = self.trade_exchange.get_factor(target, start_time, end_time)
        amount = self.trade_exchange.round_amount_by_trade_unit(amount, factor)
        if amount <= 0:
            return TradeDecisionWO([], self)
        self.buy_step = trade_step
        return TradeDecisionWO(
            [
                Order(
                    stock_id=target,
                    amount=amount,
                    direction=Order.BUY,
                    start_time=start_time,
                    end_time=end_time,
                )
            ],
            self,
        )
