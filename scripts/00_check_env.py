"""环境自检：确认整套 A股量化环境装好了。

用法：
    python scripts/00_check_env.py
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

CHECK = [
    ("numpy", "数值计算"),
    ("pandas", "数据框"),
    ("pyarrow", "parquet 读写"),
    ("akshare", "A股数据源"),
    ("matplotlib", "绘图"),
    ("vnpy", "VeighNa 交易平台内核"),
    ("vnpy_ctastrategy", "CTA 策略引擎"),
    ("vnpy_ctabacktester", "CTA 回测 GUI"),
    ("vnpy_sqlite", "SQLite 数据库适配器"),
    ("vnpy_datamanager", "数据管理模块"),
    ("vnpy_chartwizard", "K线图表模块"),
    ("vnpy_portfoliostrategy", "组合策略模块"),
    ("vnpy_riskmanager", "风险管理模块"),
    ("qlib", "微软 Qlib 量化平台"),
    ("lightgbm", "梯度提升树模型"),
]


def main() -> int:
    ok, bad = [], []
    print("=" * 58)
    print(f"Python : {sys.version.split()[0]}")
    print(f"解释器 : {sys.executable}")
    print("=" * 58)

    for mod, desc in CHECK:
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "")
            if not ver:
                v = getattr(m, "VERSION", None)
                ver = ".".join(str(x) for x in v) if isinstance(v, (tuple, list)) else "-"
            ok.append(mod)
            print(f"  [OK ] {mod:<24s} {ver:<14s} {desc}")
        except Exception as e:
            bad.append((mod, f"{type(e).__name__}: {e}"))
            print(f"  [FAIL] {mod:<24s} {'':<14s} {desc}  <- {type(e).__name__}")

    print("=" * 58)
    import quantdata as qd

    stocks = qd.cached_codes()
    idx = qd.cached_codes(is_index=True)
    qlib_dir = qd.ROOT / "qlib_data"
    cal = qlib_dir / "calendars" / "day.txt"
    inst = qlib_dir / "instruments" / "all.txt"

    print(f"项目根目录 : {qd.ROOT}")
    print(f"行情缓存   : {len(stocks)} 只股票, {len(idx)} 个指数 -> {qd.DAILY_DIR}")
    if stocks:
        df = qd.load_daily(stocks[0])
        if df is not None:
            print(f"           样例 {stocks[0]}: {len(df)} 行, "
                  f"{df['date'].min():%Y-%m-%d} ~ {df['date'].max():%Y-%m-%d}")
    print(f"Qlib 数据  : {'已生成' if cal.exists() else '未生成'} "
          f"({cal.read_text(encoding='utf-8').count(chr(10)) if cal.exists() else 0} 个交易日, "
          f"{inst.read_text(encoding='utf-8').count(chr(10)) if inst.exists() else 0} 个标的)")

    print("=" * 58)
    if bad:
        print(f"自检结果：{len(ok)} 项通过，{len(bad)} 项失败")
        for mod, err in bad:
            print(f"  - {mod}: {err}")
        return 1
    print(f"自检结果：全部 {len(ok)} 项通过，环境就绪。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
