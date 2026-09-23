"""第六步：信号有效性验证 —— 判断「模型到底有没有 edge」，而不只是看净值。

为什么需要这一步：回测净值很容易骗人。
  - IC≈0 的模型也可能因为「池子风格溢价」「数据错位」跑出漂亮的超额；
  - 反之 IC 低也不代表无效（Alpha158 的中周期因子在日频上 IC 天然很低）。

所以这里用**三条互相独立的证据**判真伪：
  1) 多周期 IC：真实信号应随持有期单调变化（而不是一团噪声）
  2) 分位组合：按 score 分 10 组，收益应单调递增、多空有明显价差
  3) 独立复现：**绕开 Qlib 撮合引擎**，用纯 pandas 重算 Topk 组合净值，
     并与「等权全池」「随机信号 Topk」两个对照比 —— 三者对得上才可信

用法：
    python scripts/06_signal_check.py --qlib-dir qlib_data_hs300 --tag _hs300
    python scripts/06_signal_check.py --qlib-dir qlib_data_csi500 --tag _csi500 --topk 50
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
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

# 与 03 脚本保持一致的 Alpha158 官方 LightGBM 调参
LGB_PARAMS = dict(
    loss="mse", colsample_bytree=0.8879, learning_rate=0.0421, subsample=0.8789,
    lambda_l1=205.6999, lambda_l2=580.9768, max_depth=8, num_leaves=210,
    num_threads=8, early_stopping_rounds=50, num_boost_round=600,
)
HORIZONS = [1, 2, 5, 10, 20]


def adaptive_segments(qlib_dir: Path) -> dict:
    """与 03 脚本同一套切分（60/20/20），必须一致才能和回测数字对齐。"""
    cal = pd.to_datetime(pd.read_csv(qlib_dir / "calendars" / "day.txt", header=None)[0])
    n = len(cal)
    a, b = int(n * 0.6), int(n * 0.8)
    d = lambda i: cal.iloc[i].strftime("%Y-%m-%d")  # noqa: E731
    return {"train": (d(0), d(a - 1)), "valid": (d(a), d(b - 1)), "test": (d(b), d(-3))}


def cum_stats(r: pd.Series) -> tuple[float, float, float]:
    n = len(r)
    if not n:
        return float("nan"), float("nan"), float("nan")
    tot = float((1 + r).prod())
    ann = tot ** (252 / n) - 1
    ir = float(r.mean() / r.std() * np.sqrt(252)) if r.std() else float("nan")
    return tot, ann, ir


def pure_portfolio(sig: pd.DataFrame, ret: pd.DataFrame, topk: int,
                   shift_sig: int = 0,
                   tradable: pd.DataFrame | None = None) -> tuple[pd.Series, pd.Series]:
    """「每日全换仓」组合：每天按 sig 选 topk 等权，吃 ret 的当期收益。

    注意：这是信号强度的**上界**，不是 Qlib 的回测口径。它每天把 topk 全部重选，
    日换手 ≈ 100%；而 Qlib 的 TopkDropoutStrategy(n_drop=k) 每天只换 k 只，
    日换手 ≈ k/topk。**两者不是同一个策略，不能直接对照**（曾经因为这个
    把「复现区间对不上」误判成 bug）。要对照引擎请用 topk_dropout()。
    """
    idx = sig.index.intersection(ret.index)
    daily = []
    used = []
    for i, dt in enumerate(idx):
        j = i + shift_sig
        if j >= len(idx):
            break
        tgt = idx[j]
        s, r = sig.loc[dt], ret.loc[tgt]
        m = s.notna() & r.notna()
        if tradable is not None and tgt in tradable.index:
            ok = tradable.loc[tgt]
            m &= s.index.isin(ok[ok].index)
        if m.sum() < topk:
            continue
        pick = s[m].nlargest(topk).index
        daily.append(float(r[pick].mean()))
        used.append(tgt)
    r = pd.Series(daily, index=used)
    return (1 + r).cumprod(), r


def topk_dropout(sig: pd.DataFrame, ret: pd.DataFrame, topk: int, n_drop: int,
                 shift_sig: int = 0, open_cost: float = 0.0005,
                 close_cost: float = 0.0015,
                 tradable: pd.DataFrame | None = None) -> pd.DataFrame:
    """忠实复现 Qlib `TopkDropoutStrategy` 的换仓规则，用于和引擎口径对照。

    规则：首日买入分数最高的 topk；此后每天把**持仓中分数最低的 n_drop 只**卖掉，
    换成**未持仓里分数最高的 n_drop 只**。日单边换手 = n_drop / topk。
    成本按 Qlib 的费率扣：单边换手 × (open_cost + close_cost)。

    tradable：布尔面板，「该股当天是否可成交」。**这个参数很关键**——Qlib 的
    `limit_threshold=0.095` 会拒绝涨停买入/跌停卖出，而小中盘动量策略选出来的
    股票经常当天涨停。不建模这个约束，复现出来的收益会明显虚高（实测中证500
    虚高 6%~22%），因为它白吃了「买不进去的涨停板」后面的涨幅。
    """
    idx = sig.index.intersection(ret.index)
    held: list = []
    recs = []
    for i, dt in enumerate(idx):
        j = i + shift_sig
        if j >= len(idx):
            break
        tgt = idx[j]
        s = sig.loc[dt].dropna()
        r = ret.loc[tgt]
        if len(s) < topk:
            continue
        ta = tradable.loc[tgt] if (tradable is not None and tgt in tradable.index) else None

        if not held:
            cand = s.index.intersection(ta[ta].index) if ta is not None else s.index
            held = list(s.reindex(cand).dropna().nlargest(topk).index)
            turn = 1.0
        else:
            held = [c for c in held if c in s.index]          # 剔除已无数据的
            if len(held) > topk:
                held = list(s.reindex(held).dropna().nlargest(topk).index)
            nd = min(n_drop, len(held))
            # 只能卖「当天可成交」的（跌停卖不掉就得继续拿着）
            sellable = list(ta.reindex(held).dropna().loc[lambda x: x].index) if ta is not None else held
            sell = list(s.reindex(sellable).dropna().sort_values().index[:nd]) if nd else []
            keep = [c for c in held if c not in set(sell)]
            pool = s.drop(labels=keep, errors="ignore")
            if ta is not None:
                pool = pool.reindex(pool.index.intersection(ta[ta].index)).dropna()
            buy = list(pool.nlargest(nd).index) if nd else []
            held = keep + buy
            turn = len(buy) / max(topk, 1)
        vals = r.reindex(held)
        gross = float(vals.mean()) if vals.notna().any() else 0.0
        recs.append((tgt, gross - turn * (open_cost + close_cost), turn, gross))
    out = pd.DataFrame(recs, columns=["date", "net", "turnover", "gross"]).set_index("date")
    out["cum_net"] = (1 + out["net"]).cumprod()
    return out


def main():
    ap = argparse.ArgumentParser(description="信号有效性验证（多周期 IC + 分位 + 独立复现）")
    ap.add_argument("--qlib-dir", default="qlib_data_hs300")
    ap.add_argument("--benchmark", default="SH000300")
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--n-drop", type=int, default=3,
                    help="必须与 03 脚本一致，否则复现口径对不上引擎")
    ap.add_argument("--limit", type=float, default=0.095,
                    help="涨跌停约束（默认 0.095，须与 03 的 limit_threshold 一致）")
    ap.add_argument("--tag", default="", help="产物文件名后缀")
    ap.add_argument("--seeds", type=int, default=10, help="随机对照的种子数")
    args = ap.parse_args()

    tag = args.tag
    qlib_dir = qd.ROOT / args.qlib_dir
    if not (qlib_dir / "calendars" / "day.txt").exists():
        print(f"找不到 Qlib 数据：{qlib_dir}，请先跑 scripts/02_to_qlib.py")
        return 1

    import qlib
    from qlib.constant import REG_CN
    from qlib.data import D

    qlib.init(provider_uri=str(qlib_dir), region=REG_CN)
    seg = adaptive_segments(qlib_dir)
    t0, t1 = seg["test"]
    codes = D.list_instruments(instruments=D.instruments("all"), as_list=True)
    print(f"Qlib 数据源 : {qlib_dir}")
    print(f"股票池      : {len(codes)} 只")
    print(f"测试期      : {t0} ~ {t1}")

    # ---------------- 训练模型拿 score ----------------
    from qlib.contrib.data.handler import Alpha158
    from qlib.contrib.model.gbdt import LGBModel
    from qlib.data.dataset import DatasetH

    handler = Alpha158(instruments="all", start_time=seg["train"][0], end_time=t1,
                       fit_start_time=seg["train"][0], fit_end_time=seg["train"][1])
    dataset = DatasetH(handler=handler, segments=seg)
    print("\n训练 LightGBM ...")
    model = LGBModel(**LGB_PARAMS)
    model.fit(dataset)
    pred = model.predict(dataset, segment="test")
    pred.name = "score"
    score = pred.unstack(level="instrument")
    print(f"score 面板: {score.shape}")
    # 落盘 score：复现逻辑/涨跌停约束调参时可以复用，省得每次重训 LightGBM
    score.to_csv(qd.ROOT / "logs" / f"score{tag}.csv", encoding="utf-8-sig")

    # ---------------- 收盘价面板（多留一段余量给远期收益） ----------------
    cal = pd.to_datetime(pd.read_csv(qlib_dir / "calendars" / "day.txt", header=None)[0])
    pad_start = cal[max(0, int(len(cal) * 0.8) - 40)].strftime("%Y-%m-%d")
    close = D.features(codes, ["$close"], start_time=pad_start, end_time=t1, freq="day")["$close"]
    close = close.unstack(level="instrument")

    # ---------------- 1. 多周期 IC ----------------
    print("\n===== 1. 多周期 Rank IC（score vs 未来 h 日收益）=====")
    ic_rows = []
    for h in HORIZONS:
        fwd = close.shift(-h) / close - 1
        ics = []
        for dt in score.index:
            if dt not in fwd.index:
                continue
            s, f = score.loc[dt], fwd.loc[dt]
            m = s.notna() & f.notna()
            if m.sum() > 5:
                ics.append(s[m].corr(f[m], method="spearman"))
        ics = pd.Series(ics).dropna()
        ic_rows.append({"h": h, "ic": float(ics.mean()),
                        "icir": float(ics.mean() / ics.std()) if ics.std() else float("nan"),
                        "days": int(len(ics))})
        print(f"  h={h:>2}日   IC={ics.mean():+.4f}   ICIR={ics.mean()/ics.std():+.4f}   天数={len(ics)}")
    mono = all(ic_rows[i]["ic"] <= ic_rows[i + 1]["ic"] for i in range(len(ic_rows) - 1))
    print(f"  → IC 随持有期单调递增: {'是 ✓（真实信号的典型特征）' if mono else '否 ✗（需警惕噪声）'}")

    # ---------------- 2. 分位组合 ----------------
    h = 5
    print(f"\n===== 2. 分位组合（按 score 分 10 组，{h} 日持有等权）=====")
    fwd = close.shift(-h) / close - 1
    buckets = {i: [] for i in range(10)}
    for dt in score.index:
        if dt not in fwd.index:
            continue
        s, f = score.loc[dt], fwd.loc[dt]
        m = s.notna() & f.notna()
        if m.sum() < 50:
            continue
        s, f = s[m], f[m]
        try:
            q = pd.qcut(s.rank(method="first"), 10, labels=False)
        except Exception:
            continue
        for i in range(10):
            sub = f[q == i]
            if len(sub):
                buckets[i].append(float(sub.mean()))
    qm = [float(np.mean(v)) if v else float("nan") for v in buckets.values()]
    for i, v in enumerate(qm):
        tagq = "低" if i == 0 else ("高" if i == 9 else "  ")
        print(f"  Q{i:d}({tagq})  {v*100:+.4f}%")
    spread = qm[9] - qm[0]
    qmono = all(qm[i] <= qm[i + 1] for i in range(9))
    print(f"  多空(Q9-Q0) = {spread*100:+.4f}% / {h}日"
          f"   严格单调: {'是 ✓' if qmono else '否（有局部交叉，看整体趋势）'}")

    # ---------------- 3. 独立复现（绕开 Qlib 撮合引擎） ----------------
    retA = close.shift(-1) / close - 1                    # T 收盘买、T+1 收盘卖
    retB = close.shift(-2) / close.shift(-1) - 1          # T+1 买、T+2 卖（= 标签口径）
    # 可成交面板：Qlib 的 limit_threshold 会拒绝涨停买入/跌停卖出，必须一起建模
    dret = close.pct_change()
    tradable = (dret.abs() < args.limit) & close.notna() & dret.notna()

    print("\n===== 3. 独立复现（纯 pandas，绕开 Qlib 撮合引擎）=====")
    print(f"  [主] TopkDropout 同规则复现：topk={args.topk}, n_drop={args.n_drop}，"
          f"已扣 Qlib 同款费率 + 涨跌停约束({args.limit:.1%})")
    repro = {}
    for label, ret, sh in [("口径A T买T+1卖", retA, 0),
                           ("口径B T+1买T+2卖(标签口径)", retB, 1)]:
        o = topk_dropout(score, ret, args.topk, args.n_drop, shift_sig=sh, tradable=tradable)
        tot, ann, ir = cum_stats(o["net"])
        repro[label] = {"net_value": tot, "annualized": ann, "ir": ir, "days": len(o),
                        "turnover_per_day": float(o["turnover"].mean())}
        print(f"    {label:26s} 净值={tot:.4f}  年化={ann:+.2%}  IR={ir:+.4f}"
              f"  日换手={o['turnover'].mean():.3f}  天数={len(o)}")

    print("  [参考] 每日全换仓（信号强度上界，日换手≈100%；不是 Qlib 口径，别拿它和引擎比）")
    for label, ret, sh in [("口径A", retA, 0), ("口径B", retB, 1)]:
        cum, r = pure_portfolio(score, ret, args.topk, shift_sig=sh, tradable=tradable)
        tot, ann, ir = cum_stats(r)
        print(f"    {label:26s} 净值={tot:.4f}  年化={ann:+.2%}  IR={ir:+.4f}")

    # 对照：等权全池
    ew = retA.mean(axis=1).dropna()
    ew = ew[(ew.index >= pd.Timestamp(t0)) & (ew.index <= pd.Timestamp(t1))]
    tot_ew, ann_ew, ir_ew = cum_stats(ew)
    print(f"  [对照] {'等权全池':26s} 净值={tot_ew:.4f}  年化={ann_ew:+.2%}  IR={ir_ew:+.4f}  天数={len(ew)}")

    # 对照：随机信号跑同一 TopkDropout 规则（多种子）
    rng = np.random.default_rng(0)
    rand_tot = []
    for _ in range(args.seeds):
        rs = pd.DataFrame(rng.random(score.shape), index=score.index, columns=score.columns)
        o = topk_dropout(rs, retA, args.topk, args.n_drop, shift_sig=0, tradable=tradable)
        if len(o):
            rand_tot.append(float(o["cum_net"].iloc[-1]))
    if rand_tot:
        print(f"  [对照] {'随机信号 TopkDropout':26s} 净值均值={np.mean(rand_tot):.4f}  "
              f"范围 [{min(rand_tot):.4f}, {max(rand_tot):.4f}]  （种子数 {len(rand_tot)}）")

    # 对照：基准
    bm = D.features([args.benchmark], ["$close"], start_time=t0, end_time=t1, freq="day")["$close"]
    bench_ret = bm.droplevel(1).pct_change().dropna()
    tot_b, ann_b, ir_b = cum_stats(bench_ret)
    print(f"  {args.benchmark + ' 基准（对照）':28s} 净值={tot_b:.4f}  年化={ann_b:+.2%}  IR={ir_b:+.4f}")

    # 读 03 脚本跑出来的引擎口径，直接对照
    mfile = qd.ROOT / "logs" / f"qlib_metrics{tag}.json"
    engine_nv = None
    if mfile.exists():
        engine_nv = json.loads(mfile.read_text(encoding="utf-8"))["net_value"]["strategy"]
        lo, hi = min(repro[k]["net_value"] for k in repro), max(repro[k]["net_value"] for k in repro)
        print(f"\n  Qlib 引擎口径净值 = {engine_nv:.4f}   独立复现区间 [{lo:.4f}, {hi:.4f}]")
        # 判定要**单向**看。复现省略了限价单排队、成交额约束、停牌期间的调仓延后等摩擦，
        # 所以它天然是「信号强度上界」：偏低是正常的，也说明引擎更保守。
        # 只有「引擎显著高于复现上界」才值得警惕 —— 那意味着引擎造出了复现拿不到的收益。
        if engine_nv > hi * 1.01:
            print(f"  ✗ 引擎净值比复现上界 {hi:.4f} 还高 {(engine_nv/hi - 1):.2%}，**需排查**"
                  f"（引擎可能凭空造收益：回溯偏差 / 撮合过松 / 数据错位）")
        elif lo <= engine_nv <= hi:
            print("  ✓ 引擎值落在复现区间内，口径高度一致")
        else:
            print(f"  ✓ 引擎值低于复现上界 {(1 - engine_nv/hi):.2%}，符合预期"
                  f"（复现未建模全部摩擦，其值本就应偏高）")

    # ---------------- 落盘 ----------------
    out = {
        "qlib_dir": args.qlib_dir, "tag": tag or "default", "benchmark": args.benchmark,
        "pool_size": len(codes), "topk": args.topk, "n_drop": args.n_drop,
        "segment": {k: list(v) for k, v in seg.items()},
        "multi_period_ic": ic_rows, "ic_monotonic": bool(mono),
        "decile_5d": qm, "decile_spread_5d": spread, "decile_monotonic": bool(qmono),
        "independent_repro_topkdropout": repro,
        "equal_weight_pool": {"net_value": tot_ew, "annualized": ann_ew, "ir": ir_ew},
        "random_topkdropout": {"mean": float(np.mean(rand_tot)), "min": float(min(rand_tot)),
                               "max": float(max(rand_tot)), "seeds": len(rand_tot)} if rand_tot else None,
        "benchmark_net_value": tot_b,
        "engine_net_value": engine_nv,
        "engine_below_repro_upper_bound": (
            None if engine_nv is None else
            bool(engine_nv <= max(repro[k]["net_value"] for k in repro) * 1.01)
        ),
    }
    (qd.ROOT / "logs" / f"signal_check{tag}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
        axes[0].plot([r["h"] for r in ic_rows], [r["ic"] for r in ic_rows],
                     marker="o", color="#D62728", lw=1.8)
        axes[0].axhline(0, color="#7F7F7F", lw=0.8)
        axes[0].set_title("多周期 Rank IC（应随持有期单调）", fontsize=12)
        axes[0].set_xlabel("持有期（交易日）")
        axes[0].set_ylabel("Rank IC")
        axes[0].grid(alpha=0.25)
        axes[1].bar(range(10), [v * 100 for v in qm], color="#1F77B4")
        axes[1].axhline(0, color="#7F7F7F", lw=0.8)
        axes[1].set_title(f"分位组合收益（Q0 最低分 → Q9 最高分，{h} 日）", fontsize=12)
        axes[1].set_xlabel("score 分位")
        axes[1].set_ylabel("平均收益 (%)")
        axes[1].set_xticks(range(10))
        axes[1].grid(alpha=0.25, axis="y")
        plt.tight_layout()
        png = qd.ROOT / "logs" / f"signal_check{tag}.png"
        plt.savefig(png, dpi=130)
        print(f"\n信号验证图: {png}")
    except Exception as e:
        print(f"[warn] 出图失败: {type(e).__name__}: {e}")

    print(f"明细 JSON : {qd.ROOT / 'logs' / f'signal_check{tag}.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
