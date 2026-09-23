"""第一步：用 akshare 下载 A 股日线数据到本地 parquet。

用法：
    python scripts/01_download_data.py                     # 沪深300 前 50 只，2018 至今
    python scripts/01_download_data.py --index 000905 --size 500   # 中证500 全池
    python scripts/01_download_data.py --index 000852 --size 1000  # 中证1000 全池
    python scripts/01_download_data.py --codes 600519,000858
    python scripts/01_download_data.py --start 2015-01-01 --force
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import quantdata as qd  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="下载 A 股日线数据（akshare）")
    ap.add_argument("--size", type=int, default=qd.DEFAULT_SIZE, help="股票池数量")
    ap.add_argument("--index", type=str, default=qd.DEFAULT_INDEX,
                    help="股票池所用的指数代码，" + "/".join(f"{k}({v[0]})" for k, v in qd.INDEX_SPECS.items()))
    ap.add_argument("--codes", type=str, default="", help="自定义股票代码，逗号分隔")
    ap.add_argument("--codes-file", type=str, default="",
                    help="从名单文件读股票池（一行一个代码），跳过网络请求。"
                         "相对路径按 data/universes/ 解析，例如 --codes-file 000905.txt")
    ap.add_argument("--start", type=str, default=qd.DEFAULT_START, help="起始日期")
    ap.add_argument("--end", type=str, default=None, help="结束日期，默认今天")
    ap.add_argument("--force", action="store_true", help="忽略缓存重新下载")
    ap.add_argument("--no-index", action="store_true", help="不用指数成分股，用内置池")
    ap.add_argument("--benchmark", type=str, default="",
                    help="额外下载的基准指数，默认与 --index 相同")
    ap.add_argument("--source", type=str, default="",
                    help="指定数据源，逗号分隔，默认 eastmoney,sina,tencent 依次降级")
    args = ap.parse_args()

    sources = [s.strip() for s in args.source.split(",") if s.strip()] or list(qd.DEFAULT_SOURCES)

    if args.codes:
        codes = [c.strip().zfill(6) for c in args.codes.split(",") if c.strip()]
    elif args.codes_file:
        codes = qd.get_universe(size=0, codes_file=args.codes_file)
    else:
        codes = qd.get_universe(args.size, index=args.index, use_index=not args.no_index)

    print(f"\n开始下载 {len(codes)} 只股票：{args.start} ~ {args.end or '今天'}（前复权）")
    print(f"数据源优先级：{' > '.join(sources)}")
    stats = qd.download_universe(codes, args.start, args.end, force=args.force, sources=sources)

    # 基准指数：策略池是指数的成分股时，用同一个指数当基准才有可比性。
    # 但基准必须落在已转成 bin 的 Qlib 数据里，所以这里一并下载。
    if args.benchmark:
        bm_index = args.benchmark
    elif args.codes_file:
        # 名单文件名就是指数代码（如 000905.txt），据此推断基准
        stem = Path(args.codes_file).stem.strip()
        bm_index = stem if stem in qd.INDEX_SPECS else "000300"
    elif args.codes:
        bm_index = "000300"
    else:
        bm_index = args.index
    bm_code = qd.index_qlib_code(bm_index)
    bm_name = qd.INDEX_SPECS.get(bm_index, (bm_index,))[0]
    print(f"\n下载基准指数 {bm_name}({bm_index}) ...")
    if args.force or not (qd.INDEX_DIR / f"{bm_code}.parquet").exists():
        bm = qd.download_index(bm_index, args.start, args.end)
        if bm is not None and not bm.empty:
            qd.save_daily(bm_code, bm, is_index=True)
            print(f"  [ok] {bm_code} {len(bm)} 行")
        else:
            print("  [warn] 基准指数未取到，Qlib 回测将不使用基准对比")
    else:
        print("  [skip] 已缓存")

    pool_index = args.index if not (args.codes or args.codes_file) else "custom"
    qd.write_meta(universe=codes, index=pool_index, start=args.start, end=args.end,
                  adjust="qfq", source="akshare", benchmark=bm_code)
    total = len(qd.cached_codes())
    print(f"\n完成：新增 {stats['ok']}，跳过 {stats['skipped']}，失败 {stats['failed']}")
    print(f"本地累计股票 {total} 只 -> {qd.DAILY_DIR}")
    if not args.codes and not args.codes_file:
        print(f"股票池名单已留档 -> {qd.UNIVERSE_DIR / (args.index + '.txt')}")


if __name__ == "__main__":
    main()
