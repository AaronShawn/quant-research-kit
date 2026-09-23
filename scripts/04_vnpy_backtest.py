"""第四步：vnpy 事件驱动回测 —— A股日线双均线策略。

这条线是「执行真实性」：逐根 bar 撮合、真实的委托/成交/持仓生命周期，
和实盘用的是同一套 CtaTemplate 策略基类，回测通过的代码可以几乎原样接到实盘网关。

用法：
    python scripts/04_vnpy_backtest.py
    python scripts/04_vnpy_backtest.py --code 000858 --fast 10 --slow 60 --capital 1000000
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import quantdata as qd  # noqa: E402

from vnpy.trader.constant import Exchange, Interval  # noqa: E402
from vnpy.trader.object import BarData  # noqa: E402
from vnpy_ctastrategy import CtaTemplate  # noqa: E402
from vnpy_ctastrategy.backtesting import BacktestingEngine  # noqa: E402

# ⚠ 单位陷阱：vnpy 的 calculate_statistics 把所有收益率类字段都乘了 100，
# 也就是「百分点」而不是「比例」。源码依据 vnpy_ctastrategy/backtesting.py：
#     total_return = (end_balance / self.capital - 1) * 100
#     annual_return = total_return / total_days * self.annual_days
#     daily_return = df["return"].mean() * 100
#     return_std   = df["return"].std() * 100
#     max_ddpercent = df["ddpercent"].min()          # 已经是百分比
# 所以这里不能再套 {:.2%}（会把 0.29% 渲染成 29%）。
PCT_KEYS = {"total_return", "annual_return", "daily_return", "return_std", "max_ddpercent"}
MONEY_KEYS = {
    "capital", "end_balance", "max_drawdown", "total_net_pnl", "daily_net_pnl",
    "total_commission", "daily_commission", "total_slippage", "daily_slippage",
    "total_turnover", "daily_turnover",
}
RATIO_KEYS = {"sharpe_ratio", "ewm_sharpe", "return_drawdown_ratio", "rgr_ratio"}
INT_KEYS = {"total_days", "profit_days", "loss_days", "max_drawdown_duration", "total_trade_count"}

STAT_LABELS = {
    "start_date": "回测开始日期", "end_date": "回测结束日期",
    "total_days": "总交易日", "profit_days": "盈利交易日", "loss_days": "亏损交易日",
    "capital": "起始资金", "end_balance": "结束资金",
    "total_return": "总收益率", "annual_return": "年化收益率",
    "max_drawdown": "最大回撤(元)", "max_ddpercent": "最大回撤比例",
    "max_drawdown_duration": "最大回撤持续(日)",
    "total_net_pnl": "总盈亏(元)", "daily_net_pnl": "日均盈亏(元)",
    "total_commission": "总手续费(元)", "daily_commission": "日均手续费(元)",
    "total_slippage": "总滑点(元)", "daily_slippage": "日均滑点(元)",
    "total_turnover": "总成交额(元)", "daily_turnover": "日均成交额(元)",
    "total_trade_count": "总成交笔数", "daily_trade_count": "日均成交笔数",
    "daily_return": "日均收益率", "return_std": "日收益率标准差",
    "sharpe_ratio": "夏普比率", "ewm_sharpe": "夏普(指数加权)",
    "return_drawdown_ratio": "收益回撤比", "rgr_ratio": "RGR 比率",
}


def fmt_stat(key: str, val) -> str:
    if key in PCT_KEYS:
        return f"{val:.2f}%"
    if key in MONEY_KEYS:
        return f"{val:,.2f}"
    if key in RATIO_KEYS:
        return f"{val:.4f}"
    if key in INT_KEYS:
        return f"{int(val):,}"
    if isinstance(val, float):
        return f"{val:,.4f}"
    return str(val)


class DoubleMaStrategy(CtaTemplate):
    """双均线策略：快线上穿慢线买入，下穿卖出。"""

    author = "ashare-quant"

    fast_window = 5
    slow_window = 20
    fixed_size = 1000

    parameters = ["fast_window", "slow_window", "fixed_size"]
    variables = ["fast_ma", "slow_ma"]

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)
        self.fast_ma = 0.0
        self.slow_ma = 0.0
        self.closes: list[float] = []
        self.prev_fast = 0.0
        self.prev_slow = 0.0
        self.bar_count = 0

    def on_init(self):
        self.write_log(f"策略初始化 快线={self.fast_window} 慢线={self.slow_window} "
                       f"手数={self.fixed_size}")

    def on_start(self):
        self.write_log("策略启动")

    def on_stop(self):
        self.write_log("策略停止")

    def on_bar(self, bar: BarData):
        self.cancel_all()
        self.bar_count += 1
        self.closes.append(bar.close_price)

        if len(self.closes) < self.slow_window:
            return

        self.fast_ma = sum(self.closes[-self.fast_window:]) / self.fast_window
        self.slow_ma = sum(self.closes[-self.slow_window:]) / self.slow_window

        # 需要前一根的值才能判断「上穿/下穿」，跳过第一根有效 bar
        if self.prev_fast == 0.0:
            self.prev_fast, self.prev_slow = self.fast_ma, self.slow_ma
            return

        cross_up = self.prev_fast <= self.prev_slow and self.fast_ma > self.slow_ma
        cross_down = self.prev_fast >= self.prev_slow and self.fast_ma < self.slow_ma

        if cross_up and self.pos == 0:
            self.buy(bar.close_price, self.fixed_size)
        elif cross_down and self.pos > 0:
            self.sell(bar.close_price, abs(self.pos))

        self.prev_fast, self.prev_slow = self.fast_ma, self.slow_ma
        self.put_event()

    def on_order(self, order):
        pass

    def on_trade(self, trade):
        self.put_event()


def load_bars(code: str) -> list[BarData]:
    """把本地 parquet 日线转成 vnpy 的 BarData 列表，绕过数据库直接喂给回测引擎。"""
    df = qd.load_daily(code)
    if df is None or df.empty:
        raise FileNotFoundError(f"本地没有 {code} 的数据，请先跑 scripts/01_download_data.py")

    ex = Exchange.SSE if qd.exchange_of(code) == "SH" else Exchange.SZSE
    bars = []
    for r in df.itertuples(index=False):
        bars.append(BarData(
            symbol=code,
            exchange=ex,
            datetime=r.date.to_pydatetime(),
            interval=Interval.DAILY,
            volume=float(r.volume),
            turnover=float(r.amount),
            open_interest=0.0,
            open_price=float(r.open),
            high_price=float(r.high),
            low_price=float(r.low),
            close_price=float(r.close),
            gateway_name="BACKTESTING",
        ))
    return bars


def main():
    ap = argparse.ArgumentParser(description="vnpy 双均线事件驱动回测")
    ap.add_argument("--code", default="600519", help="股票代码，6 位")
    ap.add_argument("--fast", type=int, default=5, help="快线周期")
    ap.add_argument("--slow", type=int, default=20, help="慢线周期")
    ap.add_argument("--size", type=int, default=1000, help="每次下单股数（100 的整数倍）")
    ap.add_argument("--capital", type=float, default=1_000_000, help="初始资金")
    ap.add_argument("--rate", type=float, default=0.0005, help="手续费率（万5）")
    ap.add_argument("--slippage", type=float, default=0.01, help="滑点（元/股）")
    args = ap.parse_args()

    bars = load_bars(args.code)
    print(f"载入 {args.code} 日线 {len(bars)} 根：{bars[0].datetime:%Y-%m-%d} ~ {bars[-1].datetime:%Y-%m-%d}")

    engine = BacktestingEngine()
    engine.set_parameters(
        vt_symbol=f"{args.code}.{('SSE' if qd.exchange_of(args.code) == 'SH' else 'SZSE')}",
        interval=Interval.DAILY,
        start=bars[0].datetime,
        end=bars[-1].datetime,
        rate=args.rate,
        slippage=args.slippage,
        size=1,
        pricetick=0.01,
        capital=args.capital,
    )
    engine.add_strategy(DoubleMaStrategy, {
        "fast_window": args.fast,
        "slow_window": args.slow,
        "fixed_size": args.size,
    })

    # 关键技巧：跳过 load_data()（它会去数据库捞），直接把内存里的 bar 塞进去
    engine.history_data = bars

    print(f"开始回测：快线 {args.fast} / 慢线 {args.slow}，初始资金 {args.capital:,.0f}")
    engine.run_backtesting()
    daily_df = engine.calculate_result()
    stats = engine.calculate_statistics(daily_df, output=False)

    print("\n===== vnpy 回测统计 =====")
    for k, v in stats.items():
        print(f"{STAT_LABELS.get(k, k):<22s}: {fmt_stat(k, v):>16s}")

    # ---------------- 出图 ----------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        df = daily_df if daily_df is not None else getattr(engine, "daily_df", None)
        if df is not None and not df.empty:
            plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(df.index, df["balance"], color="#1F77B4", lw=1.6, label="账户权益")
            if "close_price" in df.columns:
                ax2 = ax.twinx()
                ax2.plot(df.index, df["close_price"], color="#999999", lw=1.0, alpha=0.7,
                         label="收盘价")
                ax2.set_ylabel("收盘价")
                ax2.legend(loc="lower right")
            ax.set_title(f"vnpy 事件驱动回测  {args.code}  双均线 {args.fast}/{args.slow}", fontsize=13)
            ax.set_ylabel("账户权益 (元)")
            ax.legend(loc="upper left")
            ax.grid(alpha=0.25)
            plt.tight_layout()
            out = qd.ROOT / "logs" / f"vnpy_backtest_{args.code}.png"
            plt.savefig(out, dpi=130)
            print(f"\n资金曲线: {out}")
    except Exception as e:
        print(f"[warn] 出图失败: {type(e).__name__}: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
