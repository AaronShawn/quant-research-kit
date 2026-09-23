"""第八步：单只股票快速体检 —— 未来 N 日价格分布 + 历史形态统计。

【这个脚本为什么不给「涨跌概率」这一个数字】

单只股票的短期方向在统计上接近不可预测（日收益自相关极弱）。任何声称能给出
「明天下跌概率 63%」的模型，绝大多数是把噪声当信号。所以这里不做方向预测，
改成给三样可检验的东西：

1. Bootstrap 分布：从该股自己的历史日收益里重抽样，模拟未来 N 日的收益分布。
   给出的是「区间 + 各分位点」，以及一个诚实的上涨概率（大概率在 45%~55% 之间，
   这恰恰说明方向不可预测）。用 bootstrap 而不是正态分布，是因为它能保留肥尾。

2. 历史形态统计：找出历史上和今天类似的形态（比如单日跌幅超过 10%），
   统计之后 N 日的实际表现。样本量会明确标出来 —— 样本少就说明结论不可信。

3. 技术位置：RSI、乖离率、距高低点回撤、波动率分位。这些是描述「现在在哪」，
   不是预测「接下来去哪」。

用法：
    python scripts\\08_stock_probe.py --hk 02513
    python scripts\\08_stock_probe.py --us MU --horizons 1,3,5,10,20
    python scripts\\08_stock_probe.py --a 600519
    python scripts\\08_stock_probe.py --crypto SOL_USDT          # Gate.io 现货日线

【加密标的与股票的三处口径差异】
  1. 一年 365 个交易日而不是 252，年化因子由 PPY 统一控制（不要写死 252）；
  2. 无涨跌停、无停牌，所以「数据缺口」是真缺口（交易所维护/下架），必须逐日按日历核对；
  3. 没有财报/估值层，L8 外部信息只能靠链上与新闻检索。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import quantdata as qd  # noqa: E402

warnings.filterwarnings("ignore")

RNG = np.random.default_rng(20260923)

# 每年交易日数：股票 252，加密 365。所有年化口径都走这个变量，不允许写死 252。
PPY = 252


# ---------------------------------------------------------------- 取数

def load_hk(symbol: str) -> pd.DataFrame:
    """港股日线。优先新浪（东财在本机被代理阻断）。"""
    import akshare as ak
    code = symbol.zfill(5)
    for fn, kw in [("stock_hk_daily", dict(symbol=code, adjust="")),
                   ("stock_hk_hist", dict(symbol=code, period="daily",
                                          start_date="20200101", end_date="20991231",
                                          adjust=""))]:
        try:
            df = getattr(ak, fn)(**kw)
            df = df.rename(columns={c: c.lower() for c in df.columns})
            df["date"] = pd.to_datetime(df["date"])
            return df.sort_values("date").reset_index(drop=True)
        except Exception as e:
            print(f"  [warn] {fn} 失败: {type(e).__name__}")
    raise RuntimeError(f"拿不到港股 {code} 的行情")


def load_us(symbol: str) -> pd.DataFrame:
    """美股日线（新浪源，本机东财被代理阻断）。数据可回溯十几年，样本量比次新股好得多。"""
    import akshare as ak
    df = ak.stock_us_daily(symbol=symbol.upper(), adjust="qfq")
    df = df.rename(columns={c: c.lower() for c in df.columns})
    df["date"] = pd.to_datetime(df["date"])
    if "amount" not in df.columns and "volume" in df.columns:
        # 新浪美股不给成交额，用 收盘价×成交量 近似（够用于流动性判断）
        df["amount"] = df["close"].astype(float) * df["volume"].astype(float)
    return df.sort_values("date").reset_index(drop=True)


def load_crypto(pair: str, bars: int = 3000, refresh: bool = False) -> pd.DataFrame:
    """Gate.io 现货日线（本机实测唯一可达的加密交易所）。

    三个必须踩对的点（见 gateio-spot-quant 技能）：
      - /spot/time 返回毫秒，from/to 必须 //1000；
      - from/to 必须对齐到 interval 边界；
      - 区间双端闭，to-from 必须 <= (limit-1)*interval；
      - 最后一根未收盘（closed=='false'）必须丢掉，否则等于偷看未来。
    """
    import requests

    cache = qd.ROOT / "data" / "crypto" / f"{pair.upper()}_1d.parquet"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists() and not refresh:
        df = pd.read_parquet(cache)
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True)

    base = "https://api.gateio.ws/api/v4"
    s = requests.Session()
    s.trust_env = False          # 走系统代理会 407/超时，Gate 必须直连
    step = 86400

    now = int(s.get(f"{base}/spot/time", timeout=10).json()["server_time"]) // 1000
    end = (now // step) * step

    rows: list[list] = []
    got = 0
    while got < bars:
        page = min(1000, bars - got)
        start = end - step * (page - 1)
        r = s.get(f"{base}/spot/candlesticks",
                  params={"currency_pair": pair.upper(), "interval": "1d",
                          "limit": page, "from": start, "to": end},
                  timeout=20)
        r.raise_for_status()
        raw = r.json()
        if not raw:
            break
        rows = raw + rows
        got += len(raw)
        new_end = int(raw[0][0]) - step
        if new_end >= end:
            break
        end = new_end
        if len(raw) < 2:
            break
        time.sleep(0.35)

    df = pd.DataFrame(rows, columns=["ts", "quote_vol", "close", "high", "low",
                                     "open", "base_vol", "closed"])
    df = df[df["closed"].astype(str).str.lower() == "true"]     # 丢未收盘那根
    df["date"] = pd.to_datetime(df["ts"].astype(int), unit="s")
    for c in ["close", "high", "low", "open", "base_vol", "quote_vol"]:
        df[c] = df[c].astype(float)
    df = df.rename(columns={"quote_vol": "amount"})
    df = (df[["date", "open", "high", "low", "close", "amount", "base_vol"]]
          .drop_duplicates("date").sort_values("date").reset_index(drop=True))
    df.to_parquet(cache)
    return df


def load_a(code: str) -> pd.DataFrame:
    """A股优先本地缓存（01 脚本下过的话）。"""
    p = qd.DAILY_DIR / f"{str(code).zfill(6)}.parquet"
    if not p.exists():
        raise RuntimeError(f"本地没有 {code}，请先跑 01_download_data.py --codes {code}")
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------- 指标

def rsi(close: pd.Series, n: int = 14) -> float:
    d = close.diff().dropna()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return float((100 - 100 / (1 + rs)).iloc[-1])


def bootstrap(r: np.ndarray, horizons: list[int], n_sim: int = 20000) -> dict:
    """从历史日收益中有放回重抽样，模拟未来 N 日累计收益分布。

    保留肥尾是选它的唯一理由：正态假设会严重低估这种股票的极端行情概率。
    """
    out = {}
    for h in horizons:
        idx = RNG.integers(0, len(r), size=(n_sim, h))
        paths = np.prod(1 + r[idx], axis=1) - 1
        out[h] = {
            "p_up": float((paths > 0).mean()),
            "p_gt_5pct": float((paths > 0.05).mean()),
            "p_lt_minus5pct": float((paths < -0.05).mean()),
            "p_lt_minus10pct": float((paths < -0.10).mean()),
            "q": {str(q): float(np.percentile(paths, q))
                  for q in [1, 5, 25, 50, 75, 95, 99]},
        }
    return out


def drift_identifiability(ret: np.ndarray) -> dict:
    """漂移到底能不能估出来？—— Merton (1980) 附录 A；Ostrov & Das (2023)。

    样本均值的标准误 = σ / √T，其中 T 是**年数**，不是观测个数。
    提高采样频率（日频换分钟频）对它毫无帮助 —— 这是最反直觉、也最容易被
    忽略的一点：多采数据只在 T 这个维度上才有用。

    它的直接后果是：短样本上「漂移」和「零」在统计上无法区分。既然分不出来，
    那"扣除多少趋势"就不是一个可以选择得更准的问题，而是在挑噪声的实现。
    """
    n = len(ret)
    T = n / PPY
    sig_a = float(ret.std() * np.sqrt(PPY))
    se = sig_a / np.sqrt(T)
    mu_a = float(ret.mean() * PPY)
    t = mu_a / se if se else float("nan")
    years_for_10pct = (sig_a / 0.10) ** 2
    return {
        "n_days": n, "years": T, "ann_vol": sig_a, "se_of_drift": se,
        "ann_drift": mu_a, "t_stat": t,
        "ci_low": mu_a - 1.96 * se, "ci_high": mu_a + 1.96 * se,
        "years_for_10pct_se": years_for_10pct,
    }


def fhs_bootstrap(ret: np.ndarray, horizons: list[int], n_sim: int = 20000,
                  dist: str = "skewt") -> dict:
    """Filtered Historical Simulation —— Barone-Adesi, Bourgoin & Giannopoulos (1998),
    "Don't look back", Risk 11(8), 100-103。Hull & White (1998) 独立提出同一思路。

    三步：
      1. 用 GJR-GARCH(1,1) 把收益拆成「条件波动率 × 近似 i.i.d. 的标准化残差」。
         选 GJR 而不是普通 GARCH，是因为股票有杠杆效应 —— 下跌推高波动率的
         力度大于上涨，这正是「最近一直跌、而且跌得越来越猛」的统计形态。
      2. 对标准化残差 z 做 bootstrap：z 已经剥掉了波动聚集，保留下来的是
         肥尾和偏度（Cont, 2001 的两条典型事实）。
      3. 用 GARCH **递归**生成未来每一天的波动率，把抽到的 z 乘回去。
         递归是关键 —— 它让模拟出来的路径自己带波动聚集，而不是简单套用
         一个固定的日波动率（那正是朴素 bootstrap 最大的毛病）。

    漂移显式设为 0（见 drift_identifiability：短样本上它不可识别）。
    保留 mean 参数仅在拟合阶段用于提取残差，生成路径时不注入。
    """
    from arch import arch_model

    r_pct = ret * 100
    am = arch_model(r_pct, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist=dist)
    res = am.fit(disp="off", show_warning=False)
    pm = res.params

    omega = float(pm["omega"])
    alpha = float(pm["alpha[1]"])
    gamma = float(pm.get("gamma[1]", 0.0))
    beta = float(pm["beta[1]"])
    mu_fit = float(pm["mu"])

    # arch 8.x 起 conditional_volatility 返回 ndarray（旧版是 Series），
    # 所以这里统一 ravel 成 ndarray，不要再调 .iloc。
    cv = np.asarray(res.conditional_volatility).ravel()
    z = (r_pct - mu_fit) / cv
    z = z[np.isfinite(z)]

    last_s2 = float(cv[-1]) ** 2
    last_e = float(r_pct[-1] - mu_fit)
    persistence = alpha + gamma / 2 + beta

    local_rng = np.random.default_rng(20260923)
    out = {}
    for h in horizons:
        zs = local_rng.choice(z, size=(n_sim, h), replace=True)
        s2 = np.full(n_sim, last_s2)
        cum = np.ones(n_sim)
        for k in range(h):
            sig = np.sqrt(np.maximum(s2, 1e-12))
            e = sig * zs[:, k]          # 漂移 = 0
            cum *= (1 + e / 100)
            neg = (e < 0).astype(float)
            s2 = omega + alpha * e ** 2 + gamma * neg * e ** 2 + beta * s2
        paths = cum - 1
        out[h] = {
            "p_up": float((paths > 0).mean()),
            "p_gt_5pct": float((paths > 0.05).mean()),
            "p_lt_minus5pct": float((paths < -0.05).mean()),
            "p_lt_minus10pct": float((paths < -0.10).mean()),
            "q": {str(q): float(np.percentile(paths, q))
                  for q in [1, 5, 25, 50, 75, 95, 99]},
        }
    out["_params"] = {"omega": omega, "alpha": alpha, "gamma": gamma, "beta": beta,
                      "dist": dist, "persistence": persistence,
                      "loglik": float(res.loglikelihood),
                      "last_sigma_ann": float(np.sqrt(last_s2) * np.sqrt(PPY)),
                      "n_obs": int(len(z))}
    return out


# ---------------------------------------------------------------- 归因

def load_bench(sym: str, asset_is_crypto: bool, asset_market: str = "A股") -> tuple[pd.DataFrame, str]:
    """把 --bench 里写的基准符号解析成日线。

    支持三类写法，自动识别：
      - ``.IXIC`` / ``.NDX`` 这类带点号的  → 新浪美股指数接口；
      - ``QQQ`` / ``COHR`` 这类纯字母      → 新浪美股（ETF 与个股同一接口）；
      - ``BTC_USDT`` 这类带下划线的        → Gate.io 加密现货（与 --crypto 同源）。

    加这一个函数是因为「没有基准就不做归因」这条规则太容易被绕过 ——
    一旦基准取不到就静默跳过，报告里就缺了 β/α 一层，读者看不出是缺失还是零。
    """
    s = sym.strip().upper()
    import akshare as ak

    if asset_is_crypto or "_" in s:
        return load_crypto(s, refresh=False), s
    # 带点号 → 新浪美股指数（.IXIC / .NDX / .DJI / .INX）
    if s.startswith("."):
        d = ak.index_us_stock_sina(symbol=s)
        d = d.rename(columns={c: c.lower() for c in d.columns})
        d["date"] = pd.to_datetime(d["date"])
        return d.sort_values("date").reset_index(drop=True), s
    # 港股指数要放在「纯字母」分支之前，否则 HSI 会被当成美股代码打到新浪美股接口上去
    if s in {"HSI", "HSCEI", "HSCCI", "TWII"}:
        d = ak.stock_hk_index_daily_sina(symbol=s)
        d = d.rename(columns={c: c.lower() for c in d.columns})
        d["date"] = pd.to_datetime(d["date"])
        return d.sort_values("date").reset_index(drop=True), s
    if s.startswith(("SH", "SZ", "BJ")) and s[2:].isdigit():
        d = ak.stock_zh_index_daily(symbol=s.lower())
        d = d.rename(columns={c: c.lower() for c in d.columns})
        d["date"] = pd.to_datetime(d["date"])
        return d.sort_values("date").reset_index(drop=True), s
    if s.isalpha():
        return load_us(s), s
    if s.isdigit():
        # 常见的裸指数代码（000300=沪深300、399006=创业板指）自动补前缀，别让用户先存 parquet
        if len(s) == 6 and (s.startswith("000") or s.startswith("399")):
            d = ak.stock_zh_index_daily(symbol=("sh" if s.startswith("000") else "sz") + s)
            d = d.rename(columns={c: c.lower() for c in d.columns})
            d["date"] = pd.to_datetime(d["date"])
            return d.sort_values("date").reset_index(drop=True), ("sh" + s if s.startswith("000") else "sz" + s).upper()
        if asset_market == "港股":
            return load_hk(s), s
        return load_a(s), s
    raise ValueError(f"无法识别基准 {s}：请用 .IXIC / QQQ / HSI / SH000300 / BTC_USDT 这种写法")


def attribution(px: pd.Series, bench: pd.Series, ppy: int = 252) -> dict:
    """相对基准的收益/波动归因：β、年化 α、相关系数、特质波动。

    没有基准就不做这一层 —— 拿一个不相关的东西当基准算出来的 β 是假的，
    所以调用方必须自己保证基准是同一资产类别的可比标的（加密就用 BTC）。
    """
    d = pd.concat([px.rename("a"), bench.rename("b")], axis=1,
                  join="inner").dropna()
    d = d[d["b"] != 0]
    ra, rb = d["a"].pct_change().dropna(), d["b"].pct_change().dropna()
    d = pd.concat([ra.rename("a"), rb.rename("b")], axis=1, join="inner").dropna()
    if len(d) < 60:
        return {"error": f"对齐后只有 {len(d)} 个样本，不足以做归因"}
    cov = np.cov(d["a"], d["b"])
    beta = float(cov[0, 1] / cov[1, 1])
    alpha = float((d["a"].mean() - beta * d["b"].mean()) * ppy)
    resid = d["a"] - beta * d["b"]
    roll = d["a"].rolling(90).corr(d["b"]).dropna()
    win = [20, 60, 120, 252]
    rs = {}
    for w in win:
        if len(d) > w:
            rs[str(w)] = {"asset": float(d["a"].tail(w).add(1).prod() - 1),
                          "bench": float(d["b"].tail(w).add(1).prod() - 1)}
    return {
        "n_aligned": int(len(d)),
        "beta": beta, "alpha_ann": alpha,
        "corr_full": float(d["a"].corr(d["b"])),
        "corr_90d_latest": float(roll.iloc[-1]) if len(roll) else float("nan"),
        "corr_90d_min": float(roll.min()) if len(roll) else float("nan"),
        "bench_vol_ann": float(d["b"].std() * np.sqrt(ppy)),
        "idio_vol_ann": float(resid.std() * np.sqrt(ppy)),
        "systematic_share": float(1 - resid.var() / d["a"].var()),
        "relative_strength": rs,
    }


# ---------------------------------------------------------------- 风险计量

def var_es(ret: np.ndarray, levels=(0.95, 0.99)) -> dict:
    """历史模拟口径的 VaR / ES（ES 即尾部条件期望，CVaR）。

    只报历史频率，不做正态假设 —— 加密资产的峰度通常在 10 以上，正态 VaR 会
    系统性低估。ES 比 VaR 更值得看：它回答「真跌穿了会跌多少」。
    """
    out = {}
    for lv in levels:
        q = float(np.percentile(ret, (1 - lv) * 100))
        tail = ret[ret <= q]
        out[str(lv)] = {"var_1d": q,
                        "es_1d": float(tail.mean()) if len(tail) else float("nan"),
                        "n_tail": int(len(tail))}
    return out


def fhs_var_es(ret: np.ndarray, horizons=(1, 5, 10), n_sim: int = 40000,
               dist: str = "skewt") -> dict:
    """FHS 口径的多周期 VaR/ES：用 GJR-GARCH 递归生成的路径，而不是单日分位乘 √h
    （√h 缩放隐含 i.i.d. 假设，与波动聚集矛盾，会低估多周期尾部）。"""
    out = {}
    try:
        f = fhs_bootstrap(ret, list(horizons), n_sim=n_sim, dist=dist)
    except Exception:
        return out
    for h in horizons:
        q = f[h]["q"]
        out[str(h)] = {"var95": q["5"], "es95": float(np.mean(
            [q["1"], q["5"]])), "var99": q["1"]}
    return out


def drawdown_stats(px: pd.Series, dates: pd.Series) -> dict:
    """回撤分析：最大回撤、当前回撤、回撤持续与恢复时长分布。"""
    peak = px.cummax()
    dd = px / peak - 1
    cur = float(dd.iloc[-1])
    imax = int(np.argmin(dd.to_numpy()))
    # 峰值日到谷底日的自然日跨度
    ipeak = int(np.argmax(px.iloc[:imax + 1].to_numpy())) if imax > 0 else 0
    under = dd < -1e-12
    # 逐段统计「在水下」的连续区间长度（自然日）
    segs, start = [], None
    u = under.to_numpy()
    for i, v in enumerate(u):
        if v and start is None:
            start = i
        elif not v and start is not None:
            segs.append((dates.iloc[start], dates.iloc[i - 1]))
            start = None
    if start is not None:
        segs.append((dates.iloc[start], dates.iloc[-1]))
    lens = [(b - a).days for a, b in segs]
    return {
        "max_drawdown": float(dd.min()),
        "max_dd_peak_date": f"{dates.iloc[ipeak]:%Y-%m-%d}",
        "max_dd_trough_date": f"{dates.iloc[imax]:%Y-%m-%d}",
        "max_dd_days_peak_to_trough": int((dates.iloc[imax] - dates.iloc[ipeak]).days),
        "max_dd_recovery_date": (f"{dates.iloc[int(np.flatnonzero(px.iloc[imax:].to_numpy() >= px.iloc[ipeak])[0]) + imax]:%Y-%m-%d}"
                                 if (px.iloc[imax:] >= px.iloc[ipeak]).any() else "至今未收复"),
        "current_drawdown": cur,
        "current_dd_days": int((dates.iloc[-1] - dates.iloc[start]).days) if start is not None else 0,
        "n_underwater_episodes": len(segs),
        "underwater_days_median": float(np.median(lens)) if lens else float("nan"),
        "underwater_days_max": float(max(lens)) if lens else float("nan"),
        "pct_time_underwater": float(under.mean()),
    }


def pattern_stat(df: pd.DataFrame, cond_name: str, mask: pd.Series,
                 horizons: list[int]) -> dict:
    """历史上满足某个形态之后，未来 N 日的实际表现。

    这里最容易骗自己的地方是**事件独立性**：同一个暴跌波段里连续触发 10 次信号，
    看着像 10 个样本，其实只是 1 个样本被数了 10 遍。所以除了去重（间隔 >3 日），
    还要按「间隔 >20 个交易日」把事件切成独立时段，并明确告诉调用方
    —— 有效样本量应该看时段数，不是事件数。
    """
    ret = df["close"].pct_change()
    # 条件收益里有多少是「什么都不做也能拿到的漂移」？必须扣掉再看超额。
    drift_d = float(ret.mean())
    hits = np.flatnonzero(mask.fillna(False).to_numpy())
    # 去掉挨在一起的重复信号，避免同一次行情被数很多遍
    dedup = [h for i, h in enumerate(hits) if i == 0 or h - hits[i - 1] > 3]
    gaps = np.diff(dedup) if len(dedup) > 1 else np.array([])
    clusters = 1 + sum(1 for g in gaps if g > 20) if dedup else 0
    res = {"cond": cond_name, "n_events": len(dedup), "n_clusters": clusters,
           "drift_daily": drift_d,
           "gap_median": float(np.median(gaps)) if len(gaps) else float("nan"),
           "gap_max": float(gaps.max()) if len(gaps) else float("nan"),
           "event_dates": [f"{df['date'].iloc[i]:%Y-%m-%d}" for i in dedup],
           "horizons": {}}
    for h in horizons:
        vals = []
        for i in dedup:
            if i + h < len(df):
                vals.append(float(df["close"].iloc[i + h] / df["close"].iloc[i] - 1))
        if vals:
            v = np.array(vals)
            drift_h = (1 + drift_d) ** h - 1
            res["horizons"][h] = {
                "n": len(v), "mean": float(v.mean()), "median": float(np.median(v)),
                "win_rate": float((v > 0).mean()),
                "worst": float(v.min()), "best": float(v.max()),
                "drift_h": float(drift_h), "excess": float(v.mean() - drift_h),
            }
    return res


# ---------------------------------------------------------------- 主流程

def main() -> int:
    ap = argparse.ArgumentParser(description="单只股票快速体检")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--hk", help="港股代码，如 02513")
    g.add_argument("--a", help="A股代码，如 600519")
    g.add_argument("--us", help="美股代码，如 MU")
    g.add_argument("--crypto", help="Gate.io 现货交易对，如 SOL_USDT")
    ap.add_argument("--horizons", default="1,3,5,10")
    ap.add_argument("--tag", default="")
    ap.add_argument("--refresh", action="store_true", help="忽略缓存重新拉取（加密用）")
    ap.add_argument("--bench", help="基准（逗号分隔多个）：.IXIC/.NDX=美股指数，QQQ/COHR=美股标的，"
                                    "BTC_USDT=加密。第一个做主 β/α，其余并列比相对强弱")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    is_crypto = bool(args.crypto)
    if is_crypto:
        globals()["PPY"] = 365          # 加密 7×24 交易，一年 365 根日线
    if args.hk:
        code, market = args.hk.zfill(5), "港股"
        df = load_hk(args.hk)
    elif args.us:
        code, market = args.us.upper(), "美股"
        df = load_us(args.us)
    elif args.crypto:
        code, market = args.crypto.upper(), "加密现货(Gate.io)"
        df = load_crypto(args.crypto, refresh=args.refresh)
    else:
        code, market = args.a.zfill(6), "A股"
        df = load_a(args.a)

    tag = args.tag or f"_{code}"
    name = f"{market} {code}"
    n = len(df)
    px = df["close"]

    print("=" * 78)
    print(f"{name}  单股体检")
    print("=" * 78)
    print(f"样本区间: {df['date'].iloc[0]:%Y-%m-%d} ~ {df['date'].iloc[-1]:%Y-%m-%d}"
          f"  共 {n} 个交易日")
    print(f"最新收盘: {px.iloc[-1]:.3f}   今日涨跌: {px.iloc[-1] / px.iloc[-2] - 1:+.2%}")

    # 样本量先摆在最前面 —— 它决定了后面所有数字的可信度
    if n < 250:
        print(f"\n[!] 只有 {n} 个交易日（不足 1 年）。历史统计的样本量严重不足，")
        print("    下面所有概率和区间的标准误都很大，只能当参考量级，不能当依据。")

    # ---------------- L1 数据质量审计 ----------------
    # 加密 7×24 交易，日历应是「每天一根」；缺一天就是真缺口（交易所维护/下架/接口丢数），
    # 不能用前向填充糊过去 —— 那会凭空造出 0 收益日，把波动率往下压。
    # 日历口径要先分清：加密是「每天一根」，股票只有工作日一根，节假日正常休市。
    # 用自然日日历去核股票会得到 1000+ 个「缺口」，全是周末 —— 假警报会把真缺口淹掉。
    print(f"\n【数据质量审计】")
    if is_crypto:
        cal = pd.date_range(df["date"].iloc[0], df["date"].iloc[-1], freq="D")
        cal_desc = "自然日（7×24 交易，缺一天就是真缺口）"
    else:
        cal = pd.bdate_range(df["date"].iloc[0], df["date"].iloc[-1])
        cal_desc = "工作日（周末休市不计缺口，余下的通常是法定节假日）"
    missing = sorted(set(cal) - set(df["date"]))
    # 把缺口切成连续段：股票假期一般是 1 天（偶尔连休），连续 3 天以上才值得怀疑
    runs, cur_run = [], []
    for i, d in enumerate(missing):
        cur_run.append(d)
        if i == len(missing) - 1 or (d - missing[i + 1]).days not in (0, -1, -3):
            runs.append(list(cur_run))
            cur_run = []
    suspicious = [r for r in runs if len(r) >= 3]
    print(f"  日历口径     : {cal_desc}")
    print(f"  日历跨度     : {(df['date'].iloc[-1] - df['date'].iloc[0]).days} 个自然日，"
          f"应有约 {len(cal)} 根日线，实收 {n} 根")
    print(f"  缺口         : {len(missing)} 个工作日"
          + (f"，分 {len(runs)} 段（正常假期级别）" if not suspicious
             else f"，其中 {len(suspicious)} 段连续 ≥3 个工作日需确认")
          + (f"  示例: {' '.join(f'{d:%Y-%m-%d}' for d in missing[:6])}"
             f"{' ...' if len(missing) > 6 else ''}" if missing else "（无）"))
    for r in suspicious[:5]:
        print(f"    [!] 连续缺口 {len(r)} 天: {r[0]:%Y-%m-%d} ~ {r[-1]:%Y-%m-%d}"
              f"（确认是真停牌还是接口丢数）")
    print(f"  缺口处理     : 保留为 NaN，不补 0、不前向填充")
    print(f"  复权         : {'加密货币不涉及复权（现货价格本身连续）' if is_crypto else '以数据源复权口径为准'}")
    dup = int(df['date'].duplicated().sum())
    print(f"  重复日期     : {dup}")
    bad = px.pct_change().abs() > 0.40
    bad_dates = " ".join(f"{df['date'].iloc[i]:%Y-%m-%d}"
                         for i in np.flatnonzero(bad.fillna(False).to_numpy())[:8])
    print(f"  单日 |涨跌|>40% : {int(bad.sum())} 天"
          + (f"  → {bad_dates}（需人工确认是真行情还是数据错误）" if bad.any() else ""))
    if is_crypto:
        print(f"  幸存者偏差   : 单标的无成分股问题；但『现在还在交易的币』本身是筛选结果，"
              f"归零的山寨币不在样本里，同类横比时会有偏")

    ret = px.pct_change().dropna()
    ann_vol = float(ret.std() * np.sqrt(PPY))
    print(f"\n【波动特征】")
    print(f"  年化口径     : 一年 {PPY} 个交易日")
    print(f"  日收益标准差   : {ret.std():.4f}   年化波动率: {ann_vol:.1%}")
    print(f"  日均收益       : {ret.mean():+.4f}  （t 值 {ret.mean() / ret.std() * np.sqrt(n):.2f}，"
          f"|t|<2 即无法拒绝「均值为零」）")
    print(f"  单日最大涨/跌  : {ret.max():+.2%} / {ret.min():+.2%}")
    print(f"  偏度/峰度      : {ret.skew():+.2f} / {ret.kurt():+.2f}"
          f"  （正态分布为 0 / 0，峰度大说明极端行情远多于正态预期）")
    print(f"  跌超 5% 的天数 : {(ret < -0.05).sum()} / {n}"
          f"   涨超 5% 的天数: {(ret > 0.05).sum()} / {n}")

    # ---------------- 技术位置 ----------------
    hi, lo = float(px.max()), float(px.min())
    print(f"\n【技术位置】（描述「现在在哪」，不是预测）")
    print(f"  区间最高/最低  : {hi:.2f} / {lo:.2f}")
    print(f"  距最高回撤     : {px.iloc[-1] / hi - 1:+.2%}")
    print(f"  位于区间位置   : {(px.iloc[-1] - lo) / (hi - lo):.1%}"
          f"  （0%=最低点，100%=最高点）")
    print(f"  RSI(14)        : {rsi(px):.1f}   （<30 超卖，>70 超买）")
    if n > 20:
        ma20 = px.rolling(20).mean().iloc[-1]
        print(f"  20 日均线乖离  : {px.iloc[-1] / ma20 - 1:+.2%}  （20日均线 {ma20:.2f}）")
    if n > 60:
        ma60 = px.rolling(60).mean().iloc[-1]
        print(f"  60 日均线乖离  : {px.iloc[-1] / ma60 - 1:+.2%}  （60日均线 {ma60:.2f}）")
    vol20 = px.pct_change().tail(20).std() * np.sqrt(PPY)
    vol60 = px.pct_change().tail(60).std() * np.sqrt(PPY) if n > 60 else np.nan
    print(f"  近 20 日年化波动: {vol20:.1%}"
          + (f"   近 60 日: {vol60:.1%}   放大 {vol20 / vol60:.2f} 倍" if vol60 == vol60 else ""))
    print(f"  当前波动率历史分位: "
          f"{(ret.rolling(20).std() * np.sqrt(PPY)).rank(pct=True).iloc[-1]:.1%}")
    if "amount" in df.columns:
        amt = df["amount"].astype(float)
        print(f"  今日成交额     : {amt.iloc[-1] / 1e8:.2f} 亿"
              f"   20 日均值: {amt.tail(20).mean() / 1e8:.2f} 亿"
              f"   放大 {amt.iloc[-1] / amt.tail(20).mean():.2f} 倍")

    # ---------------- Bootstrap ----------------
    # 三个口径一起给，因为第一个口径必然有偏：
    #   全历史  —— 含样本期自身的漂移。智谱这段恰好是 +459% 的暴涨期，
    #              直接重抽样会把「上涨」这个趋势当成常态继承下来，概率虚高。
    #   去漂移  —— 扣掉日均收益，只保留波动形状，回答「如果没有任何趋势，会怎样」。
    #   近 60 日—— 反映当前所处的波动状态，而不是半年前的。
    r_all = ret.to_numpy()
    r_recent = ret.tail(min(60, n)).to_numpy()
    boot_raw = bootstrap(r_all, horizons)
    boot_dm = bootstrap(r_all - r_all.mean(), horizons)
    boot_rec = bootstrap(r_recent, horizons)
    print(f"\n【未来 N 日收益分布】bootstrap 重抽样 20000 次")
    print(f"  {'持有':>4} {'上涨概率(全历史)':>17} {'上涨概率(去漂移)':>17} {'上涨概率(近60日)':>17}")
    for h in horizons:
        print(f"  {h:>3}日 {boot_raw[h]['p_up']:>16.1%} {boot_dm[h]['p_up']:>16.1%} "
              f"{boot_rec[h]['p_up']:>16.1%}")
    drift_gap = max(abs(boot_raw[h]["p_up"] - boot_dm[h]["p_up"]) for h in horizons)
    if drift_gap > 0.03:
        print(f"  [!] 两个口径差了 {drift_gap:.1%}，说明样本期的整体涨跌方向对概率影响很大，")
        print(f"      全历史那一列不可直接当概率用。以「去漂移」列为准更稳妥。")

    print(f"\n  {'持有':>4} {'跌超5%概率':>11} {'跌超10%概率':>12} "
          f"{'5%分位':>9} {'中位数':>9} {'95%分位':>9}   （去漂移口径）")
    for h in horizons:
        b = boot_dm[h]
        print(f"  {h:>3}日 {b['p_lt_minus5pct']:>10.1%} {b['p_lt_minus10pct']:>11.1%} "
              f"{b['q']['5']:>+9.2%} {b['q']['50']:>+9.2%} {b['q']['95']:>+9.2%}")

    last = px.iloc[-1]
    print(f"\n  换算成价格（现价 {last:.2f}，去漂移口径）:")
    print(f"  {'持有':>4} {'5%分位':>10} {'25%分位':>10} {'中位':>10} {'75%分位':>10} {'95%分位':>10}")
    for h in horizons:
        q = boot_dm[h]["q"]
        print(f"  {h:>3}日 " + " ".join(
            f"{last * (1 + q[str(p)]):>10.2f}" for p in [5, 25, 50, 75, 95]))

    print(f"\n  [!] 换成人话：能估计的是「幅度」，不是「方向」。")
    print(f"      上面这几列价格区间，才是这只股票未来几天真正可预期的信息。")

    # ---------------- 漂移到底能不能估 ----------------
    di = drift_identifiability(ret.to_numpy())
    print(f"\n【漂移到底能不能估？】Merton (1980) 附录 A；Ostrov & Das (2023)")
    print(f"  年化波动率 σ          : {di['ann_vol']:.1%}")
    print(f"  样本长度 T            : {di['years']:.3f} 年（{di['n_days']} 个交易日）")
    print(f"  漂移估计的标准误 σ/√T : {di['se_of_drift']:.1%}")
    print(f"  观测到的年化漂移      : {di['ann_drift']:+.1%}")
    print(f"  t 值                  : {di['t_stat']:.2f}"
          f"   （{'拒绝' if abs(di['t_stat']) >= 1.96 else '无法拒绝'}「漂移 = 0」）")
    print(f"  漂移的 95% 置信区间   : [{di['ci_low']:+.0%}, {di['ci_high']:+.0%}]")
    if abs(di["t_stat"]) < 1.96:
        print(f"  → 标准误比漂移估计值本身还大，区间跨了好几倍本金 —— 漂移与零不可区分。")
    else:
        print(f"  → 样本够长，漂移在统计上显著。")
    print(f"  → 要把标准误压到 10%，需要 {di['years_for_10pct_se']:.0f} 年数据；"
          f"当前已有 {di['years']:.1f} 年。")
    print(f"  → 注意：提高采样频率不会缩小这个标准误，只有拉长年数才行。")
    if abs(di["t_stat"]) < 1.96:
        print(f"  结论：漂移与零不可区分。此时任何「扣多少趋势 / 用多长窗口」的选择，")
        print(f"        都不是在挑更准的估计，而是在挑噪声的一个具体实现 ——")
        print(f"        所以最科学的处理是**承认漂移不可识别、显式设为零**，")
        print(f"        把不确定性全部交给波动率通道表达。这正是下一段 FHS 的做法。")
    else:
        # 漂移显著也不等于该把它外推到几天内：短期里漂移的贡献远小于波动。
        d_daily = di["ann_drift"] / PPY
        drift_5d = (1 + d_daily) ** 5 - 1
        vol_5d = di["ann_vol"] * np.sqrt(5 / PPY)
        print(f"  结论：漂移统计显著，但不代表该把它外推到未来几天 ——")
        print(f"        5 日漂移贡献约 {drift_5d:+.2%}，而 5 日波动 {vol_5d:.1%}，"
              f"漂移只占波动幅度的一小部分。")
        print(f"        另外它假设漂移恒定，而 Pesaran & Timmermann (2007) 指出参数会结构性突变，")
        print(f"        「过去 15 年 +{di['ann_drift']:.0%}/年」这种速度本身也依赖 AI 周期能否延续。")
        print(f"        FHS 仍然设漂移为零 —— 这是保守选择，也让区间可比。")

    # ---------------- FHS ----------------
    print(f"\n【FHS 过滤历史模拟】Barone-Adesi, Bourgoin & Giannopoulos (1998), Risk 11(8)")
    try:
        fhs = fhs_bootstrap(ret.to_numpy(), horizons)
        fp = fhs["_params"]
        print(f"  模型: GJR-GARCH(1,1), 残差分布={fp['dist']}（GJR 项捕捉「跌时波动更大」的杠杆效应）")
        print(f"  估计参数: ω={fp['omega']:.4f} α={fp['alpha']:.4f} "
              f"γ={fp['gamma']:.4f} β={fp['beta']:.4f}")
        print(f"  持续性 α+γ/2+β = {fp['persistence']:.4f}"
              f"   （趋近 1 说明波动冲击衰减很慢）")
        # 注意：last_sigma_ann 是「百分数数值」（如 160.7 表示 160.7%），
        # 不是比率，所以用 %.1f%% 而不是 %.1f%%，否则会多乘 100。
        print(f"  当前条件年化波动率: {fp['last_sigma_ann']:.1f}%"
              f"   （对比样本无条件年化 σ = {ann_vol:.1%}）")
        print(f"\n  {'持有':>4} {'上涨概率':>9} {'跌超5%':>9} {'跌超10%':>9} "
              f"{'1%分位':>9} {'5%分位':>9} {'中位':>9} {'95%分位':>9}")
        for h in horizons:
            b = fhs[h]
            print(f"  {h:>3}日 {b['p_up']:>8.1%} {b['p_lt_minus5pct']:>8.1%} "
                  f"{b['p_lt_minus10pct']:>8.1%} {b['q']['1']:>+9.2%} "
                  f"{b['q']['5']:>+9.2%} {b['q']['50']:>+9.2%} {b['q']['95']:>+9.2%}")

        h5 = 5 if 5 in horizons else horizons[-1]
        print(f"\n  与前面三个口径对比（未来 {h5} 日）:")
        print(f"    {'口径':<34} {'上涨概率':>9} {'5%分位':>10} {'95%分位':>10}")
        cmp_rows = [
            ("A 全历史 bootstrap（含样本期漂移，有偏）",
             boot_raw[h5]["p_up"], boot_raw[h5]["q"]["5"], boot_raw[h5]["q"]["95"]),
            ("B 去漂移 bootstrap（粗暴去均值）",
             boot_dm[h5]["p_up"], boot_dm[h5]["q"]["5"], boot_dm[h5]["q"]["95"]),
            (f"C 近 {min(60, n)} 日 bootstrap（窗口任选 + 自带近期趋势）",
             boot_rec[h5]["p_up"], boot_rec[h5]["q"]["5"], boot_rec[h5]["q"]["95"]),
            ("D FHS-GARCH（文献推荐）",
             fhs[h5]["p_up"], fhs[h5]["q"]["5"], fhs[h5]["q"]["95"]),
        ]
        for lab, pu, q5, q95 in cmp_rows:
            print(f"    {lab:<34} {pu:>8.1%} {q5:>+10.2%} {q95:>+10.2%}")
        print(f"\n    A 的偏差来自样本期趋势；C 的偏差来自窗口内趋势 + 选窗口的任意性；")
        print(f"    B 和 D 都设漂移为零，但 D 额外处理了波动聚集 —— 这才是两者真正的差别。")
    except Exception as e:
        fhs = None
        print(f"  [warn] FHS 失败: {type(e).__name__}: {e}")
        print(f"  （需要 pip install arch；也可能样本太短导致 GARCH 不收敛）")

    # ---------------- L7 归因 ----------------
    # 逗号分隔多个基准：第一个当主基准做 β/α，其余并列打印（同业比选用同一套口径）
    attr, attrs = None, {}
    bench_arg = (args.bench or "").strip()
    if bench_arg:
        for bi, bsym in enumerate(b.strip() for b in bench_arg.split(",") if b.strip()):
            print(f"\n【归因分解】基准 = {bsym.upper()}")
            try:
                bdf, bname = load_bench(bsym, is_crypto, market)
                # 两边都用 DatetimeIndex 再对齐：主标的是 RangeIndex，直接 concat 会全空
                bser = bdf.set_index("date")["close"].astype(float)
                a1 = attribution(px.set_axis(df["date"]), bser, PPY)
                if "error" in a1:
                    print(f"  [skip] {a1['error']}")
                    continue
                if bi == 0:
                    attr = a1
                attrs[bname] = a1
                print(f"  区间           : {bdf['date'].iloc[0]:%Y-%m-%d} ~ "
                      f"{bdf['date'].iloc[-1]:%Y-%m-%d}   对齐样本 {a1['n_aligned']} 天")
                print(f"  β              : {a1['beta']:.2f}   "
                      f"（基准动 1%，本标的平均动 {a1['beta']:.2f}%）")
                print(f"  年化 α         : {a1['alpha_ann']:+.1%}")
                print(f"  相关系数（全样本/近90日）: {a1['corr_full']:.2f} / "
                      f"{a1['corr_90d_latest']:.2f}   近90日最低 {a1['corr_90d_min']:.2f}")
                print(f"  波动拆解       : 基准年化波动 {a1['bench_vol_ann']:.1%}，"
                      f"特质波动 {a1['idio_vol_ann']:.1%}，"
                      f"系统性占比 {a1['systematic_share']:.1%}")
                print(f"  {'窗口':>6} {'本标的':>10} {'基准':>10} {'相对':>10}")
                for w, v in a1["relative_strength"].items():
                    print(f"  {w:>5}日 {v['asset']:>+9.1%} {v['bench']:>+9.1%} "
                          f"{v['asset'] - v['bench']:>+9.1%}")
            except Exception as e:
                print(f"  [warn] 归因失败({bsym}): {type(e).__name__}: {e}")
    elif not bench_arg:
        print(f"\n【归因分解】未指定基准 → 跳过 β/α/相对强弱（无基准时这些数字算不出来，"
              f"不是零）")

    # ---------------- L4 风险计量 ----------------
    print(f"\n【风险计量】VaR / ES / 回撤")
    ve = var_es(ret.to_numpy())
    for lv, d in ve.items():
        print(f"  历史口径 {lv}: 单日 VaR {d['var_1d']:+.2%}   "
              f"ES（跌穿后平均）{d['es_1d']:+.2%}   尾部样本 {d['n_tail']} 天")
    fve = fhs_var_es(ret.to_numpy(), horizons=tuple(h for h in horizons if h > 1) or (5,))
    if fve:
        print(f"  FHS+GJR-GARCH 多周期（含波动聚集，不用 √h 缩放）:")
        for h, d in fve.items():
            print(f"    {h:>3}日 VaR95 {d['var95']:+.2%}   VaR99 {d['var99']:+.2%}")
    dds = drawdown_stats(px, df["date"])
    print(f"  最大回撤     : {dds['max_drawdown']:+.2%}  "
          f"（{dds['max_dd_peak_date']} → {dds['max_dd_trough_date']}，"
          f"{dds['max_dd_days_peak_to_trough']} 天），收复于 {dds['max_dd_recovery_date']}")
    print(f"  当前回撤     : {dds['current_drawdown']:+.2%}  （已在水下 {dds['current_dd_days']} 天）")
    print(f"  水下区间     : 共 {dds['n_underwater_episodes']} 段，时长中位 "
          f"{dds['underwater_days_median']:.0f} 天 / 最长 {dds['underwater_days_max']:.0f} 天，"
          f"处于水下的时间占比 {dds['pct_time_underwater']:.1%}")
    r_sorted = ret.sort_values()
    print(f"  最差单日     : {ret.min():+.2%}   最差 5 日累计: "
          f"{float((px / px.shift(5) - 1).min()):+.2%}   最差 20 日累计: "
          f"{float((px / px.shift(20) - 1).min()):+.2%}")

    # ---------------- 历史形态 ----------------
    print(f"\n【历史类似形态之后的表现】")
    ret_full = px.pct_change()
    conds = [
        (f"单日跌幅 ≤ -10%", ret_full <= -0.10),
        (f"单日跌幅 ≤ -5%", ret_full <= -0.05),
        (f"20日跌幅 ≥ 20%", px / px.shift(20) - 1 <= -0.20),
        (f"跌破 20 日均线 5% 以上", px / px.rolling(20).mean() - 1 <= -0.05),
    ]
    pats = []
    for cname, mask in conds:
        p = pattern_stat(df, cname, mask, horizons)
        pats.append(p)
        if p["n_events"] == 0:
            print(f"\n  「{cname}」: 历史上从未出现（样本 0），无法统计")
            continue
        # 有效样本看波段数而不是事件数 —— 同一波暴跌里触发的信号不算独立样本
        eff = p["n_clusters"]
        gm, gx = p["gap_median"], p["gap_max"]
        if gm == gm and gm < 20:
            print(f"\n  「{cname}」触发 {p['n_events']} 次，但相邻间隔中位数只有 "
                  f"{gm:.0f} 个交易日 → 事件扎堆在 {eff} 个波段里")
        else:
            print(f"\n  「{cname}」触发 {p['n_events']} 次，分布在 {eff} 个独立波段")
        if eff < 8:
            print(f"    ⚠ 有效样本只有 {eff} 个波段（事件数 {p['n_events']} 是虚的），"
                  f"胜率不具备统计意义")
        ds = p["event_dates"]
        print(f"    触发日: {' '.join(ds[:12])}{' ...' if len(ds) > 12 else ''}")
        print(f"    {'持有':>4} {'胜率':>8} {'平均':>9} {'其中漂移':>9} {'超额':>9} "
              f"{'中位':>9} {'最差':>9} {'最好':>9} {'样本':>5}")
        for h in horizons:
            if h in p["horizons"]:
                s = p["horizons"][h]
                print(f"    {h:>3}日 {s['win_rate']:>7.1%} {s['mean']:>+9.2%} "
                      f"{s['drift_h']:>+9.2%} {s['excess']:>+9.2%} "
                      f"{s['median']:>+9.2%} {s['worst']:>+9.2%} {s['best']:>+9.2%} {s['n']:>5}")
        print(f"    （「其中漂移」= 全样本日均 {p['drift_daily']:+.3%} 复利 {h} 日的贡献；"
              f"平均 − 漂移 = 超额，超额才是这个形态真正带来的东西）")
    print(f"\n  [!] 读法：如果「独立时段」只有个位数，那这一行的胜率基本没有统计意义——")
    print(f"      它相当于「过去 3 次里有 2 次上涨」，你不能据此推断第 4 次。")

    # ---------------- 出图 ----------------
    png = None
    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False

            fig, axes = plt.subplots(1, 3, figsize=(17, 5.2),
                                     gridspec_kw={"width_ratios": [2.3, 1.2, 1.2]})
            tail = min(120, n)
            d = df.tail(tail)
            axes[0].plot(d["date"], d["close"], color="#C0392B", lw=1.5, label="收盘价")
            if tail > 20:
                axes[0].plot(d["date"], px.rolling(20).mean().tail(tail),
                             color="#F39C12", lw=1.1, label="20日均线")
            # 未来区间扇形
            fut_x = [d["date"].iloc[-1]]
            for h in horizons:
                fut_x.append(d["date"].iloc[-1] + pd.Timedelta(days=int(h * 1.45)))
            for p_, alpha, lab in [(95, 0.16, "5%~95%"), (75, 0.22, "25%~75%")]:
                lo_p = [last] + [last * (1 + boot_dm[h]["q"][str(100 - p_)]) for h in horizons]
                hi_p = [last] + [last * (1 + boot_dm[h]["q"][str(p_)]) for h in horizons]
                axes[0].fill_between(fut_x, lo_p, hi_p, color="#C0392B", alpha=alpha,
                                     label=f"朴素去漂移 {lab}")
            axes[0].plot(fut_x, [last] + [last * (1 + boot_dm[h]["q"]["50"]) for h in horizons],
                         color="#C0392B", lw=1.3, ls="--", label="去漂移 中位")
            # 叠加 FHS：两种方法都设漂移为零，差别在是否建模波动聚集
            if fhs:
                f_lo = [last] + [last * (1 + fhs[h]["q"]["5"]) for h in horizons]
                f_hi = [last] + [last * (1 + fhs[h]["q"]["95"]) for h in horizons]
                axes[0].fill_between(fut_x, f_lo, f_hi, color="#185FA5", alpha=0.18,
                                     label="FHS-GARCH 5%~95%")
                axes[0].plot(fut_x, [last] + [last * (1 + fhs[h]["q"]["50"]) for h in horizons],
                             color="#185FA5", lw=1.5, ls="--", label="FHS 中位")
            axes[0].set_title(f"{name} 走势与未来区间（{d['date'].iloc[0]:%m/%d} 起）")
            axes[0].legend(fontsize=8)
            axes[0].grid(alpha=0.25)
            axes[0].tick_params(axis="x", rotation=30, labelsize=8)

            h5 = 5 if 5 in horizons else horizons[-1]
            idx = RNG.integers(0, len(ret), size=(20000, h5))
            paths = np.prod(1 + ret.to_numpy()[idx], axis=1) - 1
            axes[1].hist(paths * 100, bins=70, color="#7F8C8D", alpha=0.85)
            for q_, c_ in [(5, "#2980B9"), (50, "#27AE60"), (95, "#2980B9")]:
                axes[1].axvline(np.percentile(paths, q_) * 100, color=c_, lw=1.3,
                                ls="--", label=f"{q_}%分位")
            axes[1].axvline(0, color="#C0392B", lw=1.2)
            axes[1].set_title(f"未来 {h5} 日收益分布（bootstrap）")
            axes[1].set_xlabel("累计收益 (%)")
            axes[1].legend(fontsize=8)
            axes[1].grid(alpha=0.25)

            axes[2].hist(ret * 100, bins=60, color="#8E44AD", alpha=0.85, label="日收益")
            axes[2].axvline(0, color="#C0392B", lw=1.2)
            axes[2].set_title("日收益分布（看肥尾）")
            axes[2].set_xlabel("日收益 (%)")
            axes[2].legend(fontsize=8)
            axes[2].grid(alpha=0.25)

            plt.tight_layout()
            png = qd.ROOT / "logs" / f"probe{tag}.png"
            plt.savefig(png, dpi=130)
            print(f"\n图: {png}")
        except Exception as e:
            print(f"[warn] 出图失败: {type(e).__name__}: {e}")

    # ---------------- 落盘 ----------------
    out = {
        "code": code, "market": market, "as_of": f"{df['date'].iloc[-1]:%Y-%m-%d}",
        "n_days": n, "last_close": float(px.iloc[-1]),
        "today_return": float(px.iloc[-1] / px.iloc[-2] - 1),
        "vol": {"daily_std": float(ret.std()), "annualized": ann_vol,
                "skew": float(ret.skew()), "kurtosis": float(ret.kurt())},
        "position": {"high": hi, "low": lo,
                     "drawdown_from_high": float(px.iloc[-1] / hi - 1),
                     "range_position": float((px.iloc[-1] - lo) / (hi - lo)),
                     "rsi14": rsi(px)},
        "bootstrap": {
            "note": "raw 含样本期漂移，有偏；demeaned 扣掉日均收益，推荐以它为准；recent 用近60日",
            "raw": {str(k): v for k, v in boot_raw.items()},
            "demeaned": {str(k): v for k, v in boot_dm.items()},
            "recent60": {str(k): v for k, v in boot_rec.items()},
        },
        "drift_identifiability": di,
        "fhs": {str(k): v for k, v in fhs.items()} if fhs else None,
        "drift_gap": drift_gap,
        "risk": {"var_es_hist": ve, "var_es_fhs": fve, "drawdown": dds,
                 "worst_day": float(ret.min()),
                 "worst_5d": float((px / px.shift(5) - 1).min()),
                 "worst_20d": float((px / px.shift(20) - 1).min())},
        "data_audit": {"span_days": int((df['date'].iloc[-1] - df['date'].iloc[0]).days),
                       "expected_bars": int(len(cal)), "actual_bars": int(n),
                       "missing_days": len(missing),
                       "missing_sample": [f"{d:%Y-%m-%d}" for d in missing[:20]],
                       "duplicate_dates": dup,
                       "extreme_days_gt40pct": int(bad.sum())},
        "periods_per_year": PPY,
        "is_crypto": is_crypto,
        "attribution": attr,
        "attribution_all": attrs,
        "patterns": pats,
        "disclaimer": "历史统计，非投资建议。样本量看 n_days；形态统计的有效样本看 n_clusters 而非 n_events。",
    }
    (qd.ROOT / "logs" / f"probe{tag}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"产物: logs/probe{tag}.json")

    print("\n" + "!" * 78)
    print("以上全部是历史统计与区间估计，不构成投资建议。方向不可预测，能估计的只是幅度。")
    print("!" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
