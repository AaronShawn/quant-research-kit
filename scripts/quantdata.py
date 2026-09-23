"""A股数据层：统一封装 akshare，本地 parquet 缓存，供 Qlib 与 vnpy 共用。

设计原则：只在这里碰 akshare。Qlib 转换脚本和 vnpy 回测脚本都从本地 parquet 读，
这样网络抖动不会污染回测，也方便复现。
"""
from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

# 控制台中文输出（Windows 默认 GBK 会炸）
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DAILY_DIR = DATA_DIR / "daily"
INDEX_DIR = DATA_DIR / "index"
UNIVERSE_DIR = DATA_DIR / "universes"   # 每次取的成分股名单留档，保证回测可复现
META_PATH = DATA_DIR / "meta.json"
for _d in (DATA_DIR, DAILY_DIR, INDEX_DIR, UNIVERSE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- 配置
DEFAULT_START = "2018-01-01"
DEFAULT_SIZE = 50
DEFAULT_INDEX = "000300"
DEFAULT_SOURCES = ("eastmoney", "sina", "tencent")
BENCHMARK_SYMBOL = "sh000300"   # 沪深300，作为 Qlib 回测基准
BENCHMARK_CODE = "SH000300"

# 可选股票池：指数代码 -> (中文名, akshare 新浪行情代码)
# 成分股走中证指数官网接口，指数日线走新浪接口，两者是独立通道（东财不可达也不影响）。
INDEX_SPECS = {
    "000300": ("沪深300", "sh000300"),
    "000905": ("中证500", "sh000905"),
    "000852": ("中证1000", "sh000852"),
    "000016": ("上证50", "sh000016"),
}

STD_COLS = ["date", "open", "high", "low", "close", "volume", "amount", "vwap"]

# akshare 中文列名 -> 标准列名
_COLMAP = {
    "日期": "date", "开盘": "open", "收盘": "close", "最高": "high", "最低": "low",
    "成交量": "volume", "成交额": "amount", "股票代码": "code",
}

# 沪深300 取不到时的兜底池（沪深主板/创业板/科创板大盘股）
FALLBACK_UNIVERSE = [
    "600519", "000858", "601318", "600036", "000333", "600276", "601166", "000651",
    "002415", "600030", "601888", "000001", "600887", "601012", "002594", "300750",
    "600900", "601899", "600309", "000725", "002304", "600585", "601668", "600048",
    "000002", "601398", "601288", "601988", "600016", "601998", "002027", "300059",
    "600009", "601601", "601628", "600104", "000063", "002714", "600690", "000568",
    "600745", "603259", "300760", "688981", "688111", "002475", "300124", "600031",
    "601766", "600050",
]


def exchange_of(code: str) -> str:
    """6位代码 -> 交易所后缀，用于 Qlib instrument 命名与 vnpy Exchange。"""
    code = str(code).zfill(6)
    if code[0] == "6":          # 沪市主板/科创板
        return "SH"
    if code[0] in ("0", "3"):   # 深市主板/创业板
        return "SZ"
    if code[0] in ("4", "8"):   # 北交所
        return "BJ"
    return "SH"


_QCODE_RE = re.compile(r"^(SH|SZ|BJ)\d{6}$", re.I)


def qlib_code(code: str) -> str:
    """000001 -> SZ000001（Qlib 格式）。

    已经是 SH600000 形态的直接透传——指数必须走这条，因为 000300（沪深300）会被
    exchange_of 误判成深市，而 000001 同时是上证指数和平安银行，无法靠代码推断。
    """
    code = str(code).strip().upper()
    if _QCODE_RE.match(code):
        return code
    code = code.zfill(6)
    return f"{exchange_of(code)}{code}"


# ---------------------------------------------------------------- 超时保护
def _call_with_timeout(fn, timeout: float, *args, **kwargs):
    """在守护线程里调用 fn，超时抛 TimeoutError。

    免费数据接口普遍**没有设 timeout**，网络抖动时 requests 会挂住几分钟甚至更久
    （实测拉中证500成分股时整个进程卡死 5 分钟无输出）。这里强制加一层超时，
    把「卡死」变成「失败并降级」。

    不能用 ThreadPoolExecutor：它的线程非守护，进程退出时会被 join，反而会挂在收尾。
    """
    import queue
    import threading

    q: queue.Queue = queue.Queue(maxsize=1)

    def _run():
        try:
            q.put((True, fn(*args, **kwargs)))
        except BaseException as e:      # noqa: BLE001 —— 要把异常原样带回主线程
            q.put((False, e))

    threading.Thread(target=_run, daemon=True).start()
    try:
        ok, val = q.get(timeout=timeout)
    except queue.Empty:
        raise TimeoutError(f"{getattr(fn, '__name__', fn)} 超过 {timeout:.0f}s 无响应") from None
    if not ok:
        raise val
    return val


# ---------------------------------------------------------------- 股票池
def index_qlib_code(index_code: str) -> str:
    """指数代码 -> Qlib 代码。A股主流指数都在上交所发布，一律 SH 前缀。

    必须显式指定：000300（沪深300）若按个股规则推断会被判成深市。
    """
    return f"SH{str(index_code).zfill(6)}"


def save_universe(name: str, codes: list[str]) -> Path:
    """成分股名单留档。指数每半年调仓，事后想复现某次回测，必须有这份名单。"""
    p = UNIVERSE_DIR / f"{name}.txt"
    p.write_text("\n".join(codes) + "\n", encoding="utf-8")
    return p


def load_universe(name: str) -> list[str] | None:
    p = UNIVERSE_DIR / f"{name}.txt"
    if not p.exists():
        return None
    codes = [x.strip() for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    return codes or None


def get_universe(size: int = DEFAULT_SIZE, index: str = DEFAULT_INDEX,
                 use_index: bool = True, save: bool = True,
                 codes_file: str | Path | None = None) -> list[str]:
    """取股票池。优先从指定指数的成分股取，失败则回退内置池。

    index: 中证/上证指数代码，见 INDEX_SPECS
    codes_file: 直接读名单文件（一行一个代码），跳过网络请求。
                名单一旦留档就可以离线复现同一批股票——网络抖动时也更稳。
    """
    if codes_file:
        p = Path(codes_file)
        if not p.is_absolute():
            p = UNIVERSE_DIR / p
        if not p.exists():
            raise FileNotFoundError(f"名单文件不存在: {p}")
        codes = [x.strip().zfill(6) for x in
                 p.read_text(encoding="utf-8").splitlines() if x.strip()]
        codes = [c for c in codes if c[0] in ("0", "3", "6")]
        print(f"[universe] 从名单文件读入 {len(codes)} 只：{p}")
        return codes[:size] if size else codes

    index = str(index).zfill(6)
    if use_index and index in INDEX_SPECS:
        name, _ = INDEX_SPECS[index]
        try:
            import akshare as ak

            df = _call_with_timeout(ak.index_stock_cons_csindex, 45, symbol=index)
            col = next((c for c in df.columns if "成分券代码" in c), None)
            if col is None:
                col = next((c for c in df.columns if "代码" in c), None)
            if col is not None:
                codes = [str(x).strip().zfill(6) for x in df[col].tolist()]
                # 只保留沪深 A 股，剔除北交所（Qlib 侧无对应数据源，且流动性口径不同）
                codes = [c for c in codes if c[0] in ("0", "3", "6")]
                if codes:
                    print(f"[universe] {name}({index}) 成分股取到 {len(codes)} 只，"
                          f"截取前 {min(size, len(codes))} 只")
                    codes = codes[:size]
                    if save:
                        save_universe(index, codes)
                    return codes
        except Exception as e:
            print(f"[universe] {name}成分股获取失败({type(e).__name__}: {e})")
            # 网络失败时优先用上次留档的名单，比退回内置池靠谱得多
            saved = load_universe(index)
            if saved:
                print(f"[universe] 改用已留档名单 data/universes/{index}.txt（{len(saved)} 只）")
                return saved[:size]
            print("[universe] 无留档名单，回退内置池")
    print(f"[universe] 使用内置股票池，取前 {size} 只")
    return FALLBACK_UNIVERSE[:size]


# ---------------------------------------------------------------- 下载
def _normalize(df: pd.DataFrame, code: str, volume_in_lot: bool = True) -> pd.DataFrame:
    """统一成标准列。

    volume_in_lot: 个股接口(东财)的成交量单位是「手」，指数的单位已经是「股」，
    单位不一致会让 vwap 差 100 倍，必须区分。
    """
    df = df.rename(columns=_COLMAP)
    df["date"] = pd.to_datetime(df["date"])
    for c in ("open", "high", "low", "close", "volume", "amount"):
        if c not in df.columns:
            df[c] = float("nan")
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if volume_in_lot:
        df["volume"] = df["volume"] * 100          # 手 -> 股

    df = df.sort_values("date").reset_index(drop=True)
    # 指数数据没有成交额，用 volume*close 兜底（此时 vwap ≈ close，符合指数实际情况）
    df["amount"] = df["amount"].fillna(df["volume"] * df["close"])
    df["vwap"] = (df["amount"] / df["volume"]).replace(
        [float("inf"), float("-inf")], float("nan")).fillna(df["close"])

    df = df[["date"] + STD_COLS[1:]].dropna(subset=["close"]).reset_index(drop=True)
    return df


def _fetch_raw(code: str, source: str, s: str, e: str, adjust: str) -> pd.DataFrame:
    """按数据源拉原始日线。s/e 为 YYYYMMDD。"""
    import akshare as ak

    code = str(code).zfill(6)
    if source == "eastmoney":
        # 东方财富：成交额/复权都最全，但 push2his 接口会被部分网络环境拦截
        return ak.stock_zh_a_hist(symbol=code, period="daily",
                                  start_date=s, end_date=e, adjust=adjust)
    prefixed = f"{exchange_of(code).lower()}{code}"
    if source == "sina":
        # 新浪：成交量单位已经是「股」
        return ak.stock_zh_a_daily(symbol=prefixed, start_date=s, end_date=e, adjust=adjust)
    if source == "tencent":
        # 腾讯：成交量单位也是「股」，字段名与新浪不同
        return ak.stock_zh_a_hist_tx(symbol=prefixed, start_date=s, end_date=e, adjust=adjust)
    raise ValueError(f"未知数据源: {source}")


# 成交量单位是否为「手」——决定要不要 ×100 换成「股」
_LOT_VOLUME = {"eastmoney": True, "sina": False, "tencent": False}

# 连续失败多少次就判定该源本次不可用，避免对 50 只股票白打 50 次
_DEAD_AFTER = 4
_source_fail: dict[str, int] = {}
_fallback_notified: set[str] = set()


def download_stock(code: str, start: str = DEFAULT_START, end: str | None = None,
                   adjust: str = "qfq", retry: int = 2,
                   sources: list[str] | None = None) -> pd.DataFrame | None:
    """下载单只 A 股日线，多数据源自动降级。

    东财 -> 新浪 -> 腾讯。任一源连续失败 _DEAD_AFTER 次就被本次运行跳过。
    """
    end = end or datetime.now().strftime("%Y-%m-%d")
    s, e = start.replace("-", ""), end.replace("-", "")
    sources = sources or list(DEFAULT_SOURCES)
    tried = []

    for source in sources:
        if _source_fail.get(source, 0) >= _DEAD_AFTER:
            continue
        for i in range(retry):
            try:
                raw = _call_with_timeout(_fetch_raw, 40, code, source, s, e, adjust)
                if raw is None or raw.empty:
                    _source_fail[source] = _source_fail.get(source, 0) + 1
                    break
                df = _normalize(raw, code, volume_in_lot=_LOT_VOLUME[source])
                _source_fail[source] = 0
                if source != sources[0] and source not in _fallback_notified:
                    _fallback_notified.add(source)
                    print(f"    [fallback] {sources[0]} 不可用，改用 {source} 提供数据")
                return df
            except Exception as ex:
                tried.append(f"{source}:{type(ex).__name__}")
                if i == retry - 1:
                    _source_fail[source] = _source_fail.get(source, 0) + 1
                else:
                    time.sleep(1.2 * (i + 1))

    if tried:
        print(f"    [fail] {code}: {' / '.join(tried)}")
    return None


def download_index(index_code: str = DEFAULT_INDEX,
                   start: str = DEFAULT_START, end: str | None = None) -> pd.DataFrame | None:
    """下载指数日线，作为 Qlib 回测基准。

    指数日线走新浪接口，与个股成分股是独立通道，所以「东财不可达」不影响基准取数。
    """
    import akshare as ak

    index_code = str(index_code).zfill(6)
    spec = INDEX_SPECS.get(index_code)
    symbol = spec[1] if spec else f"sh{index_code}"

    for i in range(3):
        try:
            raw = _call_with_timeout(ak.stock_zh_index_daily, 30, symbol=symbol)
            raw["date"] = pd.to_datetime(raw["date"])
            raw = raw[raw["date"] >= pd.Timestamp(start)]
            if end:
                raw = raw[raw["date"] <= pd.Timestamp(end)]
            # 指数成交量单位本来就是「股」，不要再 ×100
            return _normalize(raw, index_qlib_code(index_code), volume_in_lot=False)
        except Exception as ex:
            if i == 2:
                print(f"[index] {symbol} 下载失败: {type(ex).__name__}: {ex}")
                return None
            time.sleep(1.5 * (i + 1))
    return None


def download_benchmark(start: str = DEFAULT_START, end: str | None = None) -> pd.DataFrame | None:
    """兼容旧调用：下载沪深300。"""
    return download_index("000300", start, end)


def save_daily(code: str, df: pd.DataFrame, is_index: bool = False) -> Path:
    d = INDEX_DIR if is_index else DAILY_DIR
    p = d / f"{code}.parquet"
    df.to_parquet(p, index=False)
    return p


def load_daily(code: str, is_index: bool = False) -> pd.DataFrame | None:
    d = INDEX_DIR if is_index else DAILY_DIR
    p = d / f"{code}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"])
    return df


def cached_codes(is_index: bool = False) -> list[str]:
    d = INDEX_DIR if is_index else DAILY_DIR
    return sorted(p.stem for p in d.glob("*.parquet"))


def probe_sources(code: str, start: str, end: str | None, adjust: str,
                  sources: list[str]) -> list[str]:
    """开跑前探活一次，返回本次真正可用的数据源（按原优先级排序）。

    免费接口的可用性会随网络环境变化（例如东财 push2his 在很多机房被拦），
    与其对每只股票挨个试错，不如先花十几秒探明。
    """
    end = end or datetime.now().strftime("%Y-%m-%d")
    s, e = start.replace("-", ""), end.replace("-", "")
    live = []
    for src in sources:
        try:
            raw = _call_with_timeout(_fetch_raw, 30, code, src, s, e, adjust)
            if raw is not None and not raw.empty:
                live.append(src)
                print(f"  [probe] {src:<10s} 可用 ({len(raw)} 行)")
            else:
                print(f"  [probe] {src:<10s} 无数据")
        except Exception as ex:
            print(f"  [probe] {src:<10s} 不可用 ({type(ex).__name__})")
    return live


def download_universe(codes: list[str], start: str = DEFAULT_START,
                      end: str | None = None, force: bool = False,
                      sources: list[str] | None = None) -> dict:
    """批量下载，带本地缓存与多源降级。返回统计信息。"""
    sources = list(sources or DEFAULT_SOURCES)
    ok = skipped = failed = 0
    t0 = time.time()

    pending = [c for c in codes if force or not (DAILY_DIR / f"{c}.parquet").exists()]
    skipped = len(codes) - len(pending)

    # 只在本次进程内探活一次
    if pending and not _source_fail:
        print(f"  探测数据源可用性（用 {pending[0]}）...")
        live = probe_sources(pending[0], start, end, "qfq", sources)
        for src in sources:
            if src not in live:
                _source_fail[src] = _DEAD_AFTER
        if not live:
            print("  [abort] 所有数据源都不可用，请检查网络（尤其代理设置）后重跑。")
            return {"ok": 0, "skipped": skipped, "failed": len(pending)}
        print(f"  探活完成，可用源：{live}")

    for i, code in enumerate(codes, 1):
        p = DAILY_DIR / f"{code}.parquet"
        if p.exists() and not force:
            continue

        # 全部数据源都已被判定不可用时，别继续白打
        if all(_source_fail.get(s, 0) >= _DEAD_AFTER for s in sources):
            print(f"  [abort] 所有数据源均连续失败（{', '.join(sources)}），"
                  f"已跳过剩余 {len(codes) - i + 1} 只。请检查网络后重跑。")
            failed += len(codes) - i + 1
            break

        df = download_stock(code, start, end, sources=sources)
        if df is None or df.empty:
            failed += 1
        else:
            save_daily(code, df)
            ok += 1
        if i % 5 == 0 or i == len(codes):
            print(f"  [{i}/{len(codes)}] 新增 {ok} 跳过 {skipped} 失败 {failed} "
                  f"({time.time()-t0:.0f}s)")
        time.sleep(0.2)   # 对免费接口客气一点
    return {"ok": ok, "skipped": skipped, "failed": failed}


def write_meta(**kv):
    meta = {}
    if META_PATH.exists():
        try:
            meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    meta.update(kv)
    meta["updated_at"] = datetime.now().isoformat(timespec="seconds")
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def load_all(codes: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """读取本地全部股票日线。"""
    codes = codes or cached_codes()
    out = {}
    for c in codes:
        df = load_daily(c)
        if df is not None and not df.empty:
            out[c] = df
    return out


if __name__ == "__main__":
    print("数据目录:", DATA_DIR)
    print("已缓存股票数:", len(cached_codes()))
    print("已缓存指数数:", len(cached_codes(is_index=True)))
