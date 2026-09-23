"""第三步：Qlib 端到端流程 —— Alpha158 因子 + LightGBM + TopkDropout 回测。

这是「因子研究」这条线：自动生成 158 个量价因子，训练 GBDT 预测未来收益，
按预测分选股回测，输出 Rank IC / 年化收益 / 最大回撤 / 夏普 等指标。

用法：
    python scripts/03_qlib_alpha158.py
    python scripts/03_qlib_alpha158.py --topk 50 --n-drop 5
    python scripts/03_qlib_alpha158.py --no-lgb      # 只看因子矩阵，不训练
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import quantdata as qd  # noqa: E402

warnings.filterwarnings("ignore")

# mlflow 3.x 起默认禁止文件后端（./mlruns），而 Qlib 的 workflow 实验管理正是用它，
# 不禁用就会抛 MlflowException: The filesystem tracking backend ... is in maintenance mode。
# 这里用官方给出的 opt-out 开关，省得用户手工配环境变量。
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

# Alpha158 官方 LightGBM 调参（来自 Qlib workflow_config_lightgbm_Alpha158）
LGB_PARAMS = dict(
    loss="mse", colsample_bytree=0.8879, learning_rate=0.0421, subsample=0.8789,
    lambda_l1=205.6999, lambda_l2=580.9768, max_depth=8, num_leaves=210,
    num_threads=8, early_stopping_rounds=50, num_boost_round=600,
)

# Qlib 官方 TopkDropoutStrategy 的模块路径在不同版本间变过，做兼容
try:
    from qlib.contrib.strategy import TopkDropoutStrategy
except Exception:  # pragma: no cover
    from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy


def adaptive_segments(qlib_dir: Path) -> dict:
    """按实际交易日历 60/20/20 切分训练/验证/测试。

    不写死日期，否则用户只下载两年数据时脚本会直接崩掉。
    注意：一律用 .iloc 取值。pandas 3.0 起 Series[-1] 不再退化成位置索引，
    会直接抛 KeyError（pandas 2.x 的旧写法在这里会炸）。
    """
    cal = pd.to_datetime(pd.read_csv(qlib_dir / "calendars" / "day.txt", header=None)[0])
    n = len(cal)
    if n < 300:
        raise RuntimeError(f"只有 {n} 个交易日，至少需要约 2 年数据，请先跑 01/02 脚本")
    a, b = int(n * 0.6), int(n * 0.8)

    def day(idx: int) -> str:
        return cal.iloc[idx].strftime("%Y-%m-%d")

    # 回测的结束日必须留出至少一根 bar 的余量：Qlib 撮合时要取 calendar[i+1]
    # 来给成交定价，直接用最后一天会 IndexError: index N is out of bounds。
    # 标签本身是「未来 2 日收益」，尾部本来也是 NaN，所以让出这两天没有损失。
    return {
        "train": (day(0), day(a - 1)),
        "valid": (day(a), day(b - 1)),
        "test": (day(b), day(-3)),
    }


def rank_ic(pred: pd.Series, label: pd.DataFrame) -> pd.Series:
    """按日计算 Rank IC。"""
    lab = label.copy()
    lab.columns = ["label"]
    merged = pred.to_frame("score").join(lab)
    merged = merged.dropna()
    return merged.groupby(level="datetime").apply(
        lambda g: g["score"].corr(g["label"], method="spearman") if len(g) > 5 else np.nan
    ).dropna()


def main():
    ap = argparse.ArgumentParser(description="Qlib Alpha158 + LightGBM 回测")
    ap.add_argument("--qlib-dir", default="qlib_data")
    ap.add_argument("--benchmark", default=qd.BENCHMARK_CODE,
                    help="基准指数 Qlib 代码，如 SH000300 / SH000905")
    ap.add_argument("--topk", type=int, default=30, help="持仓股票数")
    ap.add_argument("--n-drop", type=int, default=3, help="每期换出股票数")
    ap.add_argument("--capital", type=float, default=1e8, help="初始资金")
    ap.add_argument("--tag", default="", help="产物文件名后缀，例如 _csi500，便于多池对比")
    ap.add_argument("--no-lgb", action="store_true", help="跳过模型训练")
    args = ap.parse_args()

    tag = args.tag
    qlib_dir = qd.ROOT / args.qlib_dir
    if not (qlib_dir / "calendars" / "day.txt").exists():
        print(f"找不到 Qlib 数据：{qlib_dir}\n请先跑 python scripts/02_to_qlib.py")
        return 1

    import qlib
    from qlib.constant import REG_CN
    from qlib.data import D

    qlib.init(provider_uri=str(qlib_dir), region=REG_CN)
    codes = D.list_instruments(instruments=D.instruments("all"), as_list=True)
    print(f"Qlib 数据源: {qlib_dir}")
    print(f"股票池数量: {len(codes)}   样例: {codes[:5]}")
    print(f"基准指数  : {args.benchmark}")

    seg = adaptive_segments(qlib_dir)
    print(f"数据切分 -> 训练 {seg['train'][0]}~{seg['train'][1]} | "
          f"验证 {seg['valid'][0]}~{seg['valid'][1]} | 测试 {seg['test'][0]}~{seg['test'][1]}")

    # ---------------- Alpha158 因子 + 标签 ----------------
    from qlib.contrib.data.handler import Alpha158
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP

    handler = Alpha158(
        instruments="all",
        start_time=seg["train"][0],
        end_time=seg["test"][1],
        fit_start_time=seg["train"][0],
        fit_end_time=seg["train"][1],
    )
    dataset = DatasetH(handler=handler, segments=seg)

    print("正在计算 Alpha158 因子（首次约 1-3 分钟）...")
    # 注意：必须通过 DatasetH.prepare 取数，它才会触发 handler.setup_data()；
    # 直接调 handler.fetch() 会拿到未初始化的数据。
    feat_test = dataset.prepare("test", col_set="feature", data_key=DataHandlerLP.DK_I)
    print(f"测试集因子矩阵: {feat_test.shape}")
    print(f"因子样例: {list(feat_test.columns[:8])} ... 共 {feat_test.shape[1]} 个")

    if args.no_lgb:
        print("\n--no-lgb 已指定，跳过训练与回测。")
        return 0

    # ---------------- 训练 LightGBM ----------------
    from qlib.contrib.model.gbdt import LGBModel

    model = LGBModel(**LGB_PARAMS)
    print("\n训练 LightGBM ...")
    model.fit(dataset)
    pred = model.predict(dataset)
    pred.name = "score"
    print(f"预测完成: {pred.shape[0]} 条记录，日期 {pred.index.get_level_values(0).min():%Y-%m-%d} "
          f"~ {pred.index.get_level_values(0).max():%Y-%m-%d}")

    label_test = dataset.prepare("test", col_set="label", data_key=DataHandlerLP.DK_L)
    ric = rank_ic(pred, label_test)
    ic_mean, ic_std = ric.mean(), ric.std()
    icir = ic_mean / ic_std if ic_std else float("nan")
    print("\n===== 信号质量（测试集）=====")
    print(f"Rank IC 均值   : {ic_mean:>10.4f}")
    print(f"Rank IC 标准差 : {ic_std:>10.4f}")
    print(f"ICIR           : {icir:>10.4f}")
    print(f"IC > 0 占比    : {(ric > 0).mean():>10.2%}")

    try:
        imp_vals = model.model.feature_importance(importance_type="gain")
        names = list(feat_test.columns)
        # LightGBM 拿到的入参是 numpy（所以默认叫 Column_27），列序与 handler 的列序一致，
        # 长度能对上才替换名字，对不上就保留占位名，避免张冠李戴。
        idx = names if len(names) == len(imp_vals) else None
        imp = pd.Series(imp_vals, index=idx).sort_values(ascending=False)
        print("\n因子重要性 Top10（gain）:")
        for k, v in imp.head(10).items():
            print(f"  {k:16s} {v:>12,.0f}")
        imp.to_csv(qd.ROOT / "logs" / f"qlib_feature_importance{tag}.csv", encoding="utf-8-sig")
    except Exception as e:
        print(f"[warn] 因子重要性导出失败: {type(e).__name__}: {e}")

    # ---------------- 回测 ----------------
    from qlib.backtest import backtest
    from qlib.contrib.evaluate import risk_analysis

    strategy = TopkDropoutStrategy(signal=pred, topk=args.topk, n_drop=args.n_drop)
    executor_cfg = {
        "class": "SimulatorExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
    }
    exchange_kwargs = {
        "freq": "day",
        "limit_threshold": 0.095,   # A股涨跌停
        "deal_price": "close",
        "open_cost": 0.0005,        # 佣金万5
        "close_cost": 0.0015,       # 含印花税的卖出成本
        "min_cost": 5,
    }
    print(f"\n回测中（topk={args.topk}, n_drop={args.n_drop}, 初始资金 {args.capital:,.0f}）...")
    # 基准必须能在 provider 里取到 $close，否则 backtest 内部才会抛错、且报错信息不直观
    bm_probe = D.features([args.benchmark], ["$close"],
                          start_time=seg["test"][0], end_time=seg["test"][1], freq="day")
    if bm_probe.empty:
        print(f"[error] 基准 {args.benchmark} 在 {qlib_dir} 里没有数据。\n"
              f"        请用 python scripts/01_download_data.py --benchmark {args.benchmark[2:]} 下载后重跑 02。")
        return 1
    print(f"基准 {args.benchmark} 校验通过，{len(bm_probe)} 个交易日")

    portfolio_metric_dict, _ = backtest(
        start_time=seg["test"][0], end_time=seg["test"][1],
        strategy=strategy, executor=executor_cfg,
        benchmark=args.benchmark, account=args.capital,
        exchange_kwargs=exchange_kwargs,
    )
    report, _ = portfolio_metric_dict[list(portfolio_metric_dict.keys())[0]]

    # risk_analysis 返回的是一个 DataFrame（列名为 'risk'，行是各指标），
    # 直接 .items() 只会拿到那一列，必须取成 Series 再遍历。
    ra = risk_analysis(report["return"] - report["bench"], freq="day", N=252)
    ra = ra.iloc[:, 0] if isinstance(ra, pd.DataFrame) else ra
    PCT_METRICS = {"annualized_return", "max_drawdown"}
    bm_label = f"{args.benchmark}"
    print(f"\n===== 回测业绩（测试集 {len(codes)} 只股票池，超额收益 vs {bm_label}）=====")
    for k, val in ra.items():
        val = float(val)
        print(f"{k:24s}: {val:>12.2%}" if k in PCT_METRICS else f"{k:24s}: {val:>12.4f}")

    # 净值的口径要统一：账户权益已经扣掉了手续费和滑点，而 report["return"]
    # 是不含成本的日收益，两者 cumprod 出来能差好几个百分点（实测 24.49% vs 17.35%），
    # 所以策略净值一律用 account 口径，和基准才可比。
    cum_s = report["account"] / args.capital
    cum_b = (1 + report["bench"]).cumprod()
    print(f"\n策略累计净值: {cum_s.iloc[-1]:.4f}  （含手续费与滑点）")
    print(f"基准累计净值: {cum_b.iloc[-1]:.4f}")
    print(f"期末资金    : {report['account'].iloc[-1]:,.0f}")

    # 指标落盘：05_compare_pools.py 靠这些 JSON 做多股票池横向对比，
    # 否则每次都得从日志里肉眼抄数字。
    metrics = {
        "tag": tag or "default",
        "pool_size": len(codes),
        "topk": args.topk,
        "n_drop": args.n_drop,
        "benchmark": args.benchmark,
        "capital": args.capital,
        "segment": {k: list(v) for k, v in seg.items()},
        "signal": {
            "rank_ic_mean": float(ic_mean),
            "rank_ic_std": float(ic_std),
            "icir": float(icir),
            "ic_positive_ratio": float((ric > 0).mean()),
            "n_days": int(len(ric)),
        },
        "excess_vs_benchmark": {k: float(v) for k, v in ra.items()},
        "net_value": {
            "strategy": float(cum_s.iloc[-1]),
            "benchmark": float(cum_b.iloc[-1]),
            "strategy_total_return": float(cum_s.iloc[-1] - 1),
            "excess_total_return": float(cum_s.iloc[-1] - cum_b.iloc[-1]),
        },
    }
    # 因子重要性前 10，用于看不同池子里「起作用的因子」是否变了
    try:
        metrics["feature_importance_top10"] = {
            str(k): float(v) for k, v in imp.head(10).items()
        }
    except Exception:
        pass
    (qd.ROOT / "logs" / f"qlib_metrics{tag}.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------------- 出图 ----------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[2.2, 1])
        axes[0].plot(cum_s.index, cum_s.values, label="策略净值", color="#D62728", lw=1.6)
        axes[0].plot(cum_b.index, cum_b.values, label=f"{bm_label} 基准", color="#7F7F7F", lw=1.4)
        axes[0].set_title(f"Qlib Alpha158 + LightGBM   TopkDropout 回测"
                          f"（{len(codes)} 只股票池，topk={args.topk}）", fontsize=13)
        axes[0].legend(loc="upper left")
        axes[0].grid(alpha=0.25)
        axes[1].fill_between(cum_s.index, cum_s.values - cum_b.values, 0,
                             color="#D62728", alpha=0.3, label="超额收益")
        axes[1].axhline(0, color="#7F7F7F", lw=0.8)
        axes[1].set_title("超额收益（策略 - 基准）", fontsize=11)
        axes[1].legend(loc="upper left")
        axes[1].grid(alpha=0.25)
        plt.tight_layout()
        out = qd.ROOT / "logs" / f"qlib_backtest{tag}.png"
        plt.savefig(out, dpi=130)
        print(f"\n净值曲线: {out}")
    except Exception as e:
        print(f"[warn] 出图失败: {type(e).__name__}: {e}")

    report.to_csv(qd.ROOT / "logs" / f"qlib_report{tag}.csv", encoding="utf-8-sig")
    return 0


if __name__ == "__main__":
    sys.exit(main())
