"""第二步：把本地 parquet 日线转换成 Qlib 的 .bin 数据格式。

Qlib 官方仓库里的 scripts/dump_bin.py 不在 pip 包里，所以这里自己实现同一套格式，
并做一次数值回读校验。格式（与 Qlib 官方一致）：

    calendars/day.txt                     每行一个交易日 YYYY-MM-DD，升序
    instruments/all.txt                   每行  CODE<TAB>起始日<TAB>结束日
    features/<小写code>/<field>.day.bin   4字节 float32 起始下标 + N个 float32 数值

⚠️ bin 的数值数组是「按日历位置」对齐的，不是「按行号」连续的：
   第 j 个值对应 calendars/day.txt 里第 (起始下标 + j) 个交易日。
A股停牌会在区间中间留下空洞，因此每只股票必须先 reindex 到
[首日, 末日] 的完整日历切片（缺口留 NaN），否则缺口之后的价格会整体前移，
因子和标签同时被污染——这个 bug 曾让 IC≈0 的模型跑出 +30% 的假超额。

用法：
    python scripts/02_to_qlib.py
    python scripts/02_to_qlib.py --out qlib_data_cn
"""
from __future__ import annotations

import argparse
import shutil
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import quantdata as qd  # noqa: E402

FIELDS = ["open", "high", "low", "close", "volume", "vwap", "factor"]


def build_calendar(data: dict[str, pd.DataFrame]) -> list[pd.Timestamp]:
    dates = set()
    for df in data.values():
        dates.update(df["date"].tolist())
    return sorted(pd.Timestamp(d) for d in dates)


def dump_bin(data: dict[str, pd.DataFrame], out_dir: Path, fields=FIELDS,
             index_codes: frozenset[str] = frozenset()) -> dict:
    """写出 Qlib bin 数据。

    index_codes 里的标的会写进 instruments/index.txt 而不是 all.txt。
    这一点很关键：Qlib 的 `D.instruments("all")` 就是读 all.txt，指数一旦混进去，
    就会被 Alpha158 当成一只「股票」提取因子、并被 TopkDropoutStrategy 当成候选持仓。
    而回测基准（benchmark=SH000905）是直接从 features/<code>/close.day.bin 取数的，
    不依赖 instruments 列表，所以分开存放不影响基准计算。
    """
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "calendars").mkdir(parents=True)
    (out_dir / "instruments").mkdir(parents=True)
    (out_dir / "features").mkdir(parents=True)

    cal = build_calendar(data)
    cal_idx = {d: i for i, d in enumerate(cal)}
    (out_dir / "calendars" / "day.txt").write_text(
        "\n".join(d.strftime("%Y-%m-%d") for d in cal) + "\n", encoding="utf-8")

    lines, idx_lines = [], []
    filled_total = 0
    for code, df in sorted(data.items()):
        qcode = qd.qlib_code(code)
        df = df.sort_values("date").reset_index(drop=True)
        df["date"] = pd.to_datetime(df["date"])

        # ---------- 关键：按日历位置对齐，停牌缺口补 NaN ----------
        # bin 的语义是「首日下标 + 从首日起、每个日历交易日的值」，位置 = 日历下标。
        # A股停牌会在区间中间留出空洞，如果照原样连续写入，缺口之后的所有价格
        # 都会整体前移，读回来就是「张冠李戴」——因子和标签同时被污染。
        # 正确做法：把源序列 reindex 到 [首日, 末日] 的完整日历切片上，缺口留 NaN。
        # Qlib 把 NaN 视为「当日无数据 / 停牌」，回测时该股当天不可交易，正合语义。
        start_i = cal_idx[df["date"].iloc[0]]
        end_i = cal_idx[df["date"].iloc[-1]]
        span = cal[start_i:end_i + 1]                    # 该标的覆盖的完整日历切片
        n_gap = len(span) - len(df)
        filled_total += max(n_gap, 0)

        s = df.set_index("date").reindex(span)
        fdir = out_dir / "features" / qcode.lower()
        fdir.mkdir(parents=True, exist_ok=True)

        for f in fields:
            if f == "factor":
                vals = np.ones(len(span), dtype="<f4")
            else:
                # 保留 NaN(不要 nan_to_num)，它承载「停牌」信息
                vals = s[f].to_numpy(dtype="<f4")
            with open(fdir / f"{f}.day.bin", "wb") as fp:
                fp.write(struct.pack("<f", float(start_i)))
                fp.write(vals.tobytes())

        line = f"{qcode}\t{span[0]:%Y-%m-%d}\t{span[-1]:%Y-%m-%d}"
        (idx_lines if code in index_codes else lines).append(line)

    if not lines:
        raise RuntimeError("没有任何个股被写出，请检查 --universe 是否与已下载数据匹配")
    (out_dir / "instruments" / "all.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if idx_lines:
        (out_dir / "instruments" / "index.txt").write_text(
            "\n".join(idx_lines) + "\n", encoding="utf-8")
    return {"instruments": len(lines), "indexes": len(idx_lines),
            "calendar_days": len(cal), "gap_days_filled": filled_total}


def verify(out_dir: Path, data: dict[str, pd.DataFrame], cal: list[pd.Timestamp],
           sample: int = 0) -> bool:
    """从 .bin 里按「日历位置」读回来，和源数据逐点比对。

    关键：比对必须按日历下标定位，不能用「第 k 个值」。
    老版本用「首日 + 连续数组」的顺序比对，永远发现不了停牌缺口错位
    （因为在错位的坐标系里它自己也自洽），所以全量检查每只股票的收盘价。
    """
    cal_pos = {d: i for i, d in enumerate(cal)}
    codes = sorted(data.keys())
    if sample:
        codes = codes[:sample]

    bad_codes = []
    for code in codes:
        qcode = qd.qlib_code(code)
        df = data[code].sort_values("date").reset_index(drop=True)
        p = out_dir / "features" / qcode.lower() / "close.day.bin"
        if not p.exists():
            bad_codes.append((qcode, -1, "bin 缺失"))
            continue
        raw = p.read_bytes()
        start_i = int(struct.unpack("<f", raw[:4])[0])
        arr = np.frombuffer(raw[4:], dtype="<f4")

        dates = pd.to_datetime(df["date"])
        src = df["close"].to_numpy(dtype="<f4")
        # 逐日按日历位置比对
        bad = 0
        first_bad = None
        for k, dt in enumerate(dates):
            pos = cal_pos.get(dt, None)
            if pos is None:
                continue
            j = pos - start_i
            if j < 0 or j >= len(arr):
                bad += 1
                if first_bad is None:
                    first_bad = (dt, float(src[k]), None)
                continue
            if not np.isclose(arr[j], src[k], rtol=1e-5, atol=1e-4):
                bad += 1
                if first_bad is None:
                    first_bad = (dt, float(src[k]), float(arr[j]))
        date_ok = cal[start_i] == pd.Timestamp(dates.iloc[0])
        if bad or not date_ok:
            bad_codes.append((qcode, bad, first_bad))

    if bad_codes:
        print(f"  [verify] 失败 {len(bad_codes)} / {len(codes)} 只：")
        for qcode, bad, info in bad_codes[:10]:
            if isinstance(info, tuple) and len(info) == 3 and info[2] is None:
                print(f"    {qcode}: {bad} 天越界（首错位点 {info[0]:%Y-%m-%d} 源={info[1]:.2f}）")
            elif isinstance(info, tuple):
                print(f"    {qcode}: {bad} 天数值不符"
                      f"（首错位点 {info[0]:%Y-%m-%d} 源={info[1]:.2f} bin={info[2]:.2f}）")
            else:
                print(f"    {qcode}: {info}")
        return False
    print(f"  [verify] {len(codes)} 只股票收盘价逐日按日历位置比对全部一致 ✓")
    return True


def main():
    ap = argparse.ArgumentParser(description="parquet -> Qlib bin")
    ap.add_argument("--out", type=str, default="qlib_data", help="输出目录")
    ap.add_argument("--universe", type=str, default="",
                    help="只转换该股票池名单（指数代码，如 000905）；默认转换本地全部个股")
    ap.add_argument("--no-index", action="store_true", help="不写入基准指数")
    args = ap.parse_args()

    out_dir = qd.ROOT / args.out

    if args.universe:
        codes = qd.load_universe(args.universe)
        if not codes:
            print(f"找不到股票池名单 data/universes/{args.universe}.txt\n"
                  f"请先跑 python scripts/01_download_data.py --index {args.universe} --size 500")
            return 1
        avail = set(qd.cached_codes())
        missing = [c for c in codes if c not in avail]
        codes = [c for c in codes if c in avail]
        print(f"股票池 {args.universe}：名单 {len(codes) + len(missing)} 只，"
              f"本地有数据 {len(codes)} 只，缺失 {len(missing)} 只")
        if missing:
            print(f"  缺失样例（前10）: {missing[:10]}")
        data = {c: df for c in codes if (df := qd.load_daily(c)) is not None and not df.empty}
    else:
        data = qd.load_all()
    if not data:
        print("本地没有数据，请先跑 python scripts/01_download_data.py")
        return 1
    print(f"读入 {len(data)} 只股票")

    index_codes: set[str] = set()
    if not args.no_index:
        # 把本地已下载的所有指数都写进去（不止基准），benchmark 想换成中证500随时可用
        for p in sorted(qd.INDEX_DIR.glob("*.parquet")):
            bm = qd.load_daily(p.stem, is_index=True)
            if bm is not None and not bm.empty:
                # 用完整代码当 key（SH000905），qlib_code 会透传，避免被误判成深市
                data[p.stem] = bm
                index_codes.add(p.stem)
                print(f"加入基准指数 {p.stem} {len(bm)} 行")

    stats = dump_bin(data, out_dir, index_codes=frozenset(index_codes))
    print(f"写出 {stats['instruments']} 只个股 + {stats['indexes']} 个指数，"
          f"{stats['calendar_days']} 个交易日 -> {out_dir}")
    print(f"停牌缺口补 NaN 的交易日合计 {stats['gap_days_filled']} 天"
          f"（这些位置不会错位，Qlib 视其为当日不可交易）")

    cal = build_calendar(data)
    ok = verify(out_dir, data, cal)
    print("校验结果:", "通过" if ok else "失败")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
