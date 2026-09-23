"""第五步：横向对比不同股票池的 Alpha158 表现。

读 03 脚本落盘的 logs/qlib_metrics*.json 与 logs/qlib_report*.csv，
输出一张对照表 + 一张净值曲线叠加图。

用法：
    python scripts/05_compare_pools.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import quantdata as qd  # noqa: E402

warnings.filterwarnings("ignore")

# 池子的展示名（按 JSON 里的 pool_size 与 tag 推断太脆，直接按 tag 映射）
POOL_LABELS = {
    "_hs300": "沪深300 全池",
    "_csi500": "中证500 全池",
    "_csi1000": "中证1000 全池",
}
POOL_COLORS = ["#D62728", "#1F77B4", "#2E8B57", "#FF7F0E"]


def load_all_metrics() -> list[dict]:
    out = []
    for p in sorted((qd.ROOT / "logs").glob("qlib_metrics*.json")):
        tag = p.stem[len("qlib_metrics"):]
        try:
            m = json.loads(p.read_text(encoding="utf-8"))
            m["_tag"] = tag
            m["_report"] = qd.ROOT / "logs" / f"qlib_report{tag}.csv"
            out.append(m)
        except Exception as e:
            print(f"[warn] 跳过 {p.name}: {type(e).__name__}: {e}")
    return out


def fmt_table(runs: list[dict]) -> None:
    cols = [
        ("股票池", lambda r: POOL_LABELS.get(r["_tag"], r["_tag"] or "默认")),
        ("只数", lambda r: f"{r['pool_size']}"),
        ("topk", lambda r: f"{r['topk']}"),
        ("Rank IC", lambda r: f"{r['signal']['rank_ic_mean']:.4f}"),
        ("IC 标准差", lambda r: f"{r['signal']['rank_ic_std']:.4f}"),
        ("ICIR", lambda r: f"{r['signal']['icir']:.4f}"),
        ("IC>0 占比", lambda r: f"{r['signal']['ic_positive_ratio']:.1%}"),
        ("IC 天数", lambda r: f"{r['signal']['n_days']}"),
        ("超额年化", lambda r: f"{r['excess_vs_benchmark'].get('annualized_return', float('nan')):.2%}"),
        ("信息比率", lambda r: f"{r['excess_vs_benchmark'].get('information_ratio', float('nan')):.4f}"),
        ("超额最大回撤", lambda r: f"{r['excess_vs_benchmark'].get('max_drawdown', float('nan')):.2%}"),
        ("策略净值", lambda r: f"{r['net_value']['strategy']:.4f}"),
        ("基准净值", lambda r: f"{r['net_value']['benchmark']:.4f}"),
    ]
    headers = [c[0] for c in cols]
    rows = [[fn(r) for _, fn in cols] for r in runs]

    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]
    # 中文列名按 2 个字符宽度算，否则表头会错位
    widths = [w + sum(1 for ch in headers[i] if ord(ch) > 0x2E80) for i, w in enumerate(widths)]

    def line(cells: list[str], pad: str = " ") -> str:
        out = []
        for i, c in enumerate(cells):
            extra = sum(1 for ch in c if ord(ch) > 0x2E80)
            out.append(c + pad * (widths[i] - len(c) - extra))
        return "  ".join(out).rstrip()

    print("\n" + line(headers))
    print("-" * (sum(widths) + 2 * (len(widths) - 1)))
    for row in rows:
        print(line(row))


def plot_overlay(runs: list[dict]) -> Path | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[warn] matplotlib 不可用: {e}")
        return None

    curves = []
    for r in runs:
        p: Path = r["_report"]
        if not p.exists():
            print(f"[warn] 缺 {p.name}，跳过曲线")
            continue
        df = pd.read_csv(p, parse_dates=["datetime"]).set_index("datetime")
        if "account" not in df.columns:
            continue
        curves.append((POOL_LABELS.get(r["_tag"], r["_tag"] or "默认"),
                       df["account"] / r["capital"],
                       (1 + df["bench"]).cumprod(),
                       r))
    if not curves:
        return None

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), height_ratios=[2.2, 1])

    for i, (label, cs, _, r) in enumerate(curves):
        c = POOL_COLORS[i % len(POOL_COLORS)]
        ic = r["signal"]["rank_ic_mean"]
        axes[0].plot(cs.index, cs.values, label=f"{label}（IC={ic:.4f}）", color=c, lw=1.7)
    # 每个池子的基准可能不同，只画第一条以示意
    axes[0].plot(curves[0][2].index, curves[0][2].values,
                 label=f"{curves[0][3]['benchmark']} 基准", color="#7F7F7F", lw=1.3, ls="--")
    axes[0].set_title("Alpha158 + LightGBM 在不同股票池上的表现（同因子集、同模型、同参数）", fontsize=13)
    axes[0].legend(loc="upper left", fontsize=9)
    axes[0].grid(alpha=0.25)
    axes[0].axhline(1.0, color="#999999", lw=0.7)

    for i, (label, cs, cb, _) in enumerate(curves):
        c = POOL_COLORS[i % len(POOL_COLORS)]
        axes[1].plot(cs.index, cs.values - cb.values, label=label, color=c, lw=1.5)
    axes[1].axhline(0, color="#7F7F7F", lw=0.8)
    axes[1].set_title("超额收益（策略 - 各自基准）", fontsize=11)
    axes[1].legend(loc="upper left", fontsize=9)
    axes[1].grid(alpha=0.25)

    plt.tight_layout()
    out = qd.ROOT / "logs" / "pool_comparison.png"
    plt.savefig(out, dpi=130)
    return out


def main():
    runs = load_all_metrics()
    if not runs:
        print("没找到 logs/qlib_metrics*.json，请先跑 python scripts/03_qlib_alpha158.py --tag _xxx")
        return 1

    runs.sort(key=lambda r: r["signal"]["rank_ic_mean"], reverse=True)
    print(f"共 {len(runs)} 个股票池的回测结果（按 Rank IC 降序）")
    fmt_table(runs)

    print("\n各池因子重要性 Top5：")
    for r in runs:
        top = r.get("feature_importance_top10") or {}
        label = POOL_LABELS.get(r["_tag"], r["_tag"] or "默认")
        items = list(top.items())[:5]
        s = "、".join(k for k, _ in items) if items else "（未导出）"
        print(f"  {label:<14s} {s}")

    out = plot_overlay(runs)
    if out:
        print(f"\n对比曲线: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
