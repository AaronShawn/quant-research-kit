"""第七步：对最新交易日打分，输出未来 N 日的涨跌概率。

这一个脚本解决两件事，**必须区分清楚**：

1. 【统计口径｜可信】把模型分数按历史分位分组，统计样本外「未来 N 日上涨的频率」。
   这是校准过的频率概率，有样本量支撑，是真正能拿来做决策的那个数。

2. 【主观口径｜定性参考】调用智谱 GLM，把市场状态 + 候选股特征喂进去做定性研判。
   **大模型输出的百分比不是校准概率**，它是语言生成的结果，没有经过 Brier/可靠性
   校准，不能当成胜率来用。它的价值在于把你没注意到的风险点和逻辑讲出来。

为什么不能只做第 2 件：LLM 没有任何内在的概率估计机制。你问它「明天涨的概率」，
它会在语言分布里挑一个听起来合理的数字填进去 —— 同一份数据问两遍，数字能差 20 个
百分点。它的信息量主要来自 prompt 里你给它的东西，而不是它自己算出来的东西。

用法：
    :: 只跑统计口径（不需要任何密钥）
    python scripts\\07_predict.py --qlib-dir qlib_data_csi500 --benchmark SH000905 --topk 30

    :: 加上智谱定性研判（需要 API Key，去 https://open.bigmodel.cn 申请，有免费额度）
    set ZHIPU_API_KEY=你的密钥
    python scripts\\07_predict.py --qlib-dir qlib_data_csi500 --benchmark SH000905 --topk 30
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

# 与 03 脚本保持完全一致的训练参数，否则「模型还是那个模型」就成了一句空话
LGB_PARAMS = dict(
    loss="mse", colsample_bytree=0.8879, learning_rate=0.0421, subsample=0.8789,
    lambda_l1=205.6999, lambda_l2=580.9768, max_depth=8, num_leaves=210,
    num_threads=8, early_stopping_rounds=50, num_boost_round=600,
)

ZHIPU_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"


# ---------------------------------------------------------------- 工具函数

def to_display_code(qcode: str) -> str:
    """SH600519 -> 600519.SH，看着顺眼一些。"""
    if len(qcode) >= 8 and qcode[:2] in ("SH", "SZ", "BJ"):
        return f"{qcode[2:]}.{qcode[:2]}"
    return qcode


def fwd_return(close: pd.DataFrame, n: int) -> pd.DataFrame:
    """未来 n 日收益（不含今日）。"""
    return close.shift(-n) / close - 1


def bucket_table(score: pd.Series, fwd: pd.DataFrame, q: int = 10) -> pd.DataFrame:
    """把分数切成 q 档，统计每档未来收益的均值与上涨频率。

    这是整个脚本里唯一可信的「概率」来源：它数的是历史真实发生过的次数。
    """
    lab = fwd.stack(dropna=True).rename("ret")
    merge = pd.concat([score.rename("score"), lab], axis=1, join="inner").dropna()
    if merge.empty:
        return pd.DataFrame()
    # 按日做截面分档，避免不同日期的分数水平不可比
    merge["bucket"] = merge.groupby(level="datetime")["score"].transform(
        lambda s: pd.qcut(s.rank(method="first"), q, labels=False, duplicates="drop")
    )
    g = merge.groupby("bucket")["ret"]
    out = pd.DataFrame({
        "平均收益": g.mean(),
        "中位收益": g.median(),
        "上涨概率": g.apply(lambda s: (s > 0).mean()),
        "样本数": g.size(),
    })
    out.index.name = "分档(0=最看空,9=最看多)"
    return out


def cum_stats(r: pd.Series) -> tuple[float, float, float]:
    r = r.dropna()
    if r.empty:
        return float("nan"), float("nan"), float("nan")
    tot = float((1 + r).prod())
    ann = tot ** (252 / len(r)) - 1
    ir = float(r.mean() / r.std() * np.sqrt(252)) if r.std() else float("nan")
    return tot, ann, ir


# ---------------------------------------------------------------- 智谱调用

def build_prompt(market: dict, rows: list[dict], stats: dict) -> str:
    """把我们算出来的结构化事实整理成 prompt。

    刻意把「统计口径的历史胜率」放在最前面，是为了让模型在一个正确的锚点附近
    做定性判断，而不是凭空编一个数字出来。
    """
    lines = [
        "你是一位 A 股量化研究助手。下面是一次量化模型在",
        f"{market['as_of']} 收盘后给出的选股结果，请做定性研判。",
        "",
        "【重要前提】",
        "1. 你不需要、也不应该编造任何数据。所有事实都在下面给出了。",
        "2. 下面的「历史上涨概率」是真实的样本外统计结果，请把它当作基准锚点。",
        "3. 如果你要给出概率判断，必须说明它是主观判断，并给出理由和反例风险。",
        "",
        "【市场状态】",
        f"- 股票池: {market['pool']}（{market['pool_size']} 只）",
        f"- 基准指数: {market['benchmark']}",
        f"- 基准近 5 日: {market['bm_5d']:+.2%}，近 20 日: {market['bm_20d']:+.2%}，"
        f"近 60 日: {market['bm_60d']:+.2%}",
        f"- 基准 20 日年化波动率: {market['bm_vol20']:.1%}",
        f"- 池内个股 20 日收益中位数: {market['pool_med20']:+.2%}"
        f"（用来判断当前是普涨还是分化行情）",
        "",
        "【模型信号的样本外历史表现（测试集，共 "
        f"{stats['n_days']} 个交易日）】",
        f"- 日频 Rank IC 均值: {stats['ic_mean']:.4f}，ICIR: {stats['icir']:.4f}",
        f"- 分数最高一组（Q9）未来 5 日上涨概率: {stats['q9_prob5']:.1%}",
        f"- 分数最低一组（Q0）未来 5 日上涨概率: {stats['q0_prob5']:.1%}",
        f"- 池内全体股票未来 5 日上涨概率（基准锚）: {stats['all_prob5']:.1%}",
        f"- 分数最高一组未来 5 日平均收益: {stats['q9_ret5']:+.2%}",
        f"- 池内全体未来 5 日平均收益: {stats['all_ret5']:+.2%}",
        "",
        f"【今日模型打分最高的 {len(rows)} 只】",
        "格式: 代码 | 模型分(百分位) | 收盘价 | 近5日 | 近20日 | 近60日 | 20日年化波动 | 20日均成交额(亿)",
    ]
    for r in rows:
        lines.append(
            f"- {r['code']} | {r['score_pct']:.1f}% | {r['close']:.2f} | "
            f"{r['ret5']:+.2%} | {r['ret20']:+.2%} | {r['ret60']:+.2%} | "
            f"{r['vol20']:.1%} | {r['amt20']:.2f}"
        )
    lines += [
        "",
        "【请输出】",
        "1. 风格判断：这批股票整体偏向什么特征（动量/反转、高波动/低波动、大盘/中盘），"
        "用上面的数字论证，不要空谈。",
        "2. 逻辑优势：这个组合相比基准，可能赚的是什么钱？",
        "3. 主要风险：哪些地方最可能失效？结合当前市场状态说，至少说 3 条。",
        "4. 主观判断：给出你对「这个组合未来 5 个交易日跑赢基准」的概率判断，"
        "并明确写出「主观判断，非统计概率」和你的依据。不要给单只股票的涨跌预测。",
        "",
        "用中文，分小标题，控制在 600 字以内。",
    ]
    return "\n".join(lines)


def call_zhipu(api_key: str, model: str, prompt: str, timeout: int = 120) -> str:
    """调智谱 v4 接口。它兼容 OpenAI 的请求格式，所以直接用标准 chat/completions。"""
    import urllib.request

    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        ZHIPU_URL, data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


# ---------------------------------------------------------------- 主流程

def main() -> int:
    ap = argparse.ArgumentParser(description="最新交易日打分 + 未来 N 日涨跌概率")
    ap.add_argument("--qlib-dir", default="qlib_data_csi500")
    ap.add_argument("--benchmark", default="SH000905")
    ap.add_argument("--topk", type=int, default=30)
    ap.add_argument("--n-drop", type=int, default=3)
    ap.add_argument("--horizons", default="1,3,5,10", help="统计的未来持有天数，逗号分隔")
    ap.add_argument("--tag", default="", help="产物后缀")
    ap.add_argument("--zhipu-key", default=os.environ.get("ZHIPU_API_KEY", ""),
                    help="智谱 API Key，也可用环境变量 ZHIPU_API_KEY")
    ap.add_argument("--zhipu-model", default="glm-4.7-flash",
                    help="glm-4.7-flash 免费 / glm-4.7 旗舰")
    ap.add_argument("--no-llm", action="store_true", help="强制跳过智谱研判")
    args = ap.parse_args()

    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
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
    cal = pd.to_datetime(pd.read_csv(qlib_dir / "calendars" / "day.txt", header=None)[0])
    as_of = cal.iloc[-1]
    print(f"Qlib 数据源 : {qlib_dir}")
    print(f"股票池      : {len(codes)} 只")
    print(f"数据截止日  : {as_of:%Y-%m-%d}")

    # ---------------- 1. 训练（切分与 03 完全一致） ----------------
    from qlib.contrib.data.handler import Alpha158
    from qlib.contrib.model.gbdt import LGBModel
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP

    n = len(cal)
    a, b = int(n * 0.6), int(n * 0.8)

    def day(i: int) -> str:
        return cal.iloc[i].strftime("%Y-%m-%d")

    train_seg = (day(0), day(a - 1))
    valid_seg = (day(a), day(b - 1))
    test_seg = (day(b), day(-3))

    # handler 一直加载到最新交易日，这样 test 段后面那一小截（原 03 为了给撮合
    # 留余量而砍掉的 3 天）也能用来预测。归一化参数仍用训练段 fit，保证与 03 一致。
    handler = Alpha158(
        instruments="all",
        start_time=train_seg[0], end_time=day(-1),
        fit_start_time=train_seg[0], fit_end_time=train_seg[1],
    )
    segs = {"train": train_seg, "valid": valid_seg, "test": test_seg,
            "predict": (day(-1), day(-1))}
    dataset = DatasetH(handler=handler, segments=segs)
    print(f"切分        : 训练 {train_seg[0]}~{train_seg[1]} | 测试 {test_seg[0]}~{test_seg[1]}")

    print("\n计算 Alpha158 因子 ...")
    feat_test = dataset.prepare("test", col_set="feature", data_key=DataHandlerLP.DK_I)
    print(f"测试集因子矩阵: {feat_test.shape}")

    print("训练 LightGBM ...")
    model = LGBModel(**LGB_PARAMS)
    model.fit(dataset)
    pred_test = model.predict(dataset, segment="test")
    pred_test.name = "score"

    label_test = dataset.prepare("test", col_set="label", data_key=DataHandlerLP.DK_L)
    lab = label_test.copy()
    lab.columns = ["label"]
    merged = pred_test.to_frame("score").join(lab).dropna()
    ric = merged.groupby(level="datetime").apply(
        lambda g: g["score"].corr(g["label"], method="spearman") if len(g) > 5 else np.nan
    ).dropna()
    ic_mean, ic_std = float(ric.mean()), float(ric.std())
    icir = ic_mean / ic_std if ic_std else float("nan")
    print(f"样本外 Rank IC 均值 {ic_mean:.4f}   ICIR {icir:.4f}   IC>0 占比 {(ric > 0).mean():.2%}")

    # ---------------- 2. 未来收益面板 ----------------
    raw = D.features(codes, ["$close", "$volume"], start_time=train_seg[0],
                     end_time=day(-1), freq="day")
    close = raw["$close"].unstack(level="instrument")
    vol = raw["$volume"].unstack(level="instrument")

    # ---------------- 3. 统计口径：历史涨跌概率 ----------------
    print("\n" + "=" * 78)
    print("【一｜统计口径】样本外历史频率 —— 这是唯一校准过的「概率」")
    print("=" * 78)

    stat_tables, stat_summary = {}, {}
    for h in horizons:
        fwd = fwd_return(close, h)
        fwd_test = fwd.loc[fwd.index.isin(ric.index)]
        tb = bucket_table(pred_test, fwd_test, q=10)
        if tb.empty:
            continue
        stat_tables[h] = tb
        all_ret = float(fwd_test.stack().dropna().mean())
        all_prob = float((fwd_test.stack().dropna() > 0).mean())
        stat_summary[h] = {
            "all_prob": all_prob, "all_ret": all_ret,
            "q0_prob": float(tb.loc[0, "上涨概率"]), "q9_prob": float(tb.loc[9, "上涨概率"]),
            "q0_ret": float(tb.loc[0, "平均收益"]), "q9_ret": float(tb.loc[9, "平均收益"]),
            "spread": float(tb.loc[9, "平均收益"] - tb.loc[0, "平均收益"]),
        }
        print(f"\n未来 {h} 日 —— 全池上涨概率 {all_prob:.2%}，全池平均收益 {all_ret:+.3%}")
        print(f"  分数档           平均收益     上涨概率      样本数")
        for bk, row in tb.iterrows():
            print(f"  Q{int(bk)} ({'最看多' if bk == 9 else '最看空' if bk == 0 else '     '})  "
                  f"{row['平均收益']:>+9.3%}   {row['上涨概率']:>8.2%}   {int(row['样本数']):>8d}")
        print(f"  → 多空价差 Q9-Q0: {stat_summary[h]['spread']:+.3%}/持有{h}日，"
              f"概率差 {stat_summary[h]['q9_prob'] - stat_summary[h]['q0_prob']:+.2%}")

    # ---------------- 4. 对最新交易日打分 ----------------
    print("\n" + "=" * 78)
    print(f"【二｜最新信号】{as_of:%Y-%m-%d} 收盘，模型打分 Top {args.topk}")
    print("=" * 78)

    pred_last = model.predict(dataset, segment="predict")
    pred_last.name = "score"
    if pred_last.empty:
        print("[error] 最新交易日没有预测值，请检查数据是否完整")
        return 1

    s_last = pred_last.droplevel("datetime").sort_values(ascending=False)
    # 分数百分位：拿历史分布的秩来标定，而不是当日截面秩，方便判断「绝对水平」
    hist_sorted = np.sort(pred_test.values)
    pct = np.searchsorted(hist_sorted, s_last.values) / len(hist_sorted)

    c_last = close.loc[:as_of].iloc[-1] if as_of in close.index else close.iloc[-1]
    px = close.loc[:as_of]
    vol_last = vol.loc[:as_of]

    def ret_n(n_: int, s: pd.Series) -> pd.Series:
        if len(s) <= n_:
            return pd.Series(np.nan, index=s.columns)
        return s.iloc[-1] / s.iloc[-1 - n_] - 1

    r5, r20, r60 = ret_n(5, px), ret_n(20, px), ret_n(60, px)
    vol20 = px.pct_change().tail(20).std() * np.sqrt(252)
    amt20 = (px.tail(20) * vol_last.tail(20)).mean() / 1e8

    pick = s_last.head(args.topk).index.tolist()
    rows = []
    for i, code in enumerate(pick):
        rows.append({
            "code": to_display_code(code),
            "qlib_code": code,
            "rank": i + 1,
            "score": float(s_last[code]),
            "score_pct": float(pct[i] * 100),
            "close": float(c_last.get(code, np.nan)),
            "ret5": float(r5.get(code, np.nan)),
            "ret20": float(r20.get(code, np.nan)),
            "ret60": float(r60.get(code, np.nan)),
            "vol20": float(vol20.get(code, np.nan)),
            "amt20": float(amt20.get(code, np.nan)),
        })

    print(f"{'序':>3} {'代码':>11} {'模型分':>9} {'历史分位':>8} "
          f"{'收盘':>8} {'近5日':>9} {'近20日':>9} {'20日波动':>9} {'20日额(亿)':>10}")
    for r in rows[:20]:
        print(f"{r['rank']:>3} {r['code']:>11} {r['score']:>9.4f} {r['score_pct']:>7.1f}% "
              f"{r['close']:>8.2f} {r['ret5']:>+9.2%} {r['ret20']:>+9.2%} "
              f"{r['vol20']:>9.1%} {r['amt20']:>10.2f}")
    if len(rows) > 20:
        print(f"    ... 其余 {len(rows) - 20} 只见产物文件")

    # ---------------- 5. 市场状态 ----------------
    bm = D.features([args.benchmark], ["$close"], start_time=train_seg[0],
                    end_time=day(-1), freq="day")["$close"].droplevel("instrument")

    def bm_ret(days: int) -> float:
        return float(bm.iloc[-1] / bm.iloc[-1 - days] - 1) if len(bm) > days else np.nan

    market = {
        "as_of": f"{as_of:%Y-%m-%d}", "pool": args.qlib_dir, "pool_size": len(codes),
        "benchmark": args.benchmark,
        "bm_5d": bm_ret(5), "bm_20d": bm_ret(20), "bm_60d": bm_ret(60),
        "bm_vol20": float(bm.pct_change().tail(20).std() * np.sqrt(252)),
        "pool_med20": float(r20.median()),
    }
    print(f"\n市场状态: 基准近5日 {market['bm_5d']:+.2%} | 近20日 {market['bm_20d']:+.2%} | "
          f"近60日 {market['bm_60d']:+.2%} | 20日年化波动 {market['bm_vol20']:.1%}")

    # ---------------- 6. 智谱定性研判 ----------------
    llm_text, llm_err = "", ""
    if args.no_llm:
        print("\n[跳过] --no-llm 已指定")
    elif not args.zhipu_key:
        print("\n[跳过] 没有智谱 API Key，只输出统计口径。")
        print("       要加上大模型研判：set ZHIPU_API_KEY=你的密钥  （https://open.bigmodel.cn 可免费申请）")
    else:
        h5 = 5 if 5 in stat_summary else horizons[-1]
        s5 = stat_summary[h5]
        stats = {
            "n_days": int(len(ric)), "ic_mean": ic_mean, "icir": icir,
            "q9_prob5": s5["q9_prob"], "q0_prob5": s5["q0_prob"],
            "all_prob5": s5["all_prob"], "q9_ret5": s5["q9_ret"], "all_ret5": s5["all_ret"],
        }
        prompt = build_prompt(market, rows, stats)
        (qd.ROOT / "logs" / f"zhipu_prompt{tag}.txt").write_text(prompt, encoding="utf-8")
        print(f"\n调用智谱 {args.zhipu_model} 做定性研判 ...")
        try:
            llm_text = call_zhipu(args.zhipu_key, args.zhipu_model, prompt)
            print("\n" + "=" * 78)
            print("【三｜主观口径】智谱 GLM 定性研判（非统计概率，仅供参考）")
            print("=" * 78)
            print(llm_text)
        except Exception as e:
            llm_err = f"{type(e).__name__}: {e}"
            print(f"[warn] 智谱调用失败：{llm_err}")

    # ---------------- 7. 落盘 ----------------
    out = {
        "as_of": f"{as_of:%Y-%m-%d}",
        "qlib_dir": args.qlib_dir,
        "benchmark": args.benchmark,
        "pool_size": len(codes),
        "topk": args.topk,
        "signal_quality": {"rank_ic_mean": ic_mean, "icir": icir,
                           "ic_positive_ratio": float((ric > 0).mean()),
                           "n_days": int(len(ric)),
                           "segment": {"train": list(train_seg), "test": list(test_seg)}},
        "market": market,
        "probability_table": {str(h): {
            "all": {"prob": stat_summary[h]["all_prob"], "ret": stat_summary[h]["all_ret"]},
            "q0": {"prob": stat_summary[h]["q0_prob"], "ret": stat_summary[h]["q0_ret"]},
            "q9": {"prob": stat_summary[h]["q9_prob"], "ret": stat_summary[h]["q9_ret"]},
            "spread": stat_summary[h]["spread"],
            "by_bucket": stat_tables[h].reset_index().to_dict("records"),
        } for h in stat_tables},
        "top_picks": rows,
        "zhipu": {"model": args.zhipu_model, "ok": bool(llm_text),
                  "error": llm_err, "text": llm_text},
    }
    (qd.ROOT / "logs" / f"predict{tag}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    pd.DataFrame(rows).to_csv(qd.ROOT / "logs" / f"predict_picks{tag}.csv",
                              index=False, encoding="utf-8-sig")
    print(f"\n产物: logs/predict{tag}.json  logs/predict_picks{tag}.csv")

    print("\n" + "!" * 78)
    print("风控提示：以上全部为历史统计与模型输出，不构成投资建议。")
    print("  1. 成分股名单是「当前」的，回看历史存在幸存者偏差，实测收益会被系统性高估。")
    print("  2. 未考虑冲击成本、成交额约束、停牌与涨跌停的真实排队情况。")
    print("  3. 概率是历史频率，不是未来保证。样本外永远可能失效。")
    print("!" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
