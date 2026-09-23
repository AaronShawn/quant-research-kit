---
name: qlib-ashare-factor-research
description: "在本地搭 A 股/美股因子研究流水线（微软 Qlib：Alpha158 → LightGBM → TopkDropout 回测），以及排查「回测净值很漂亮但信号其实无效」这类问题时的必备知识。核心是三条：① Qlib 的 .bin 数据必须按『日历位置』写而不是按行号连续写，否则停牌缺口会让价格整体错位，把 IC 压到 0 的同时造出虚假超额（本项目实测 +30% 假超额）；② 判断信号真伪必须用『多周期 IC + 分位组合 + 绕开回测引擎的独立复现 + 随机信号对照』四件套，只看净值和日频 IC 会被骗；③ 超额收益里常常混着『等权 vs 市值加权』的风格溢价，跨股票池比较要看 IC/ICIR 而不是超额。触发词：Qlib、qlib、Alpha158、dump_bin、bin 格式、Rank IC、ICIR、因子回测、TopkDropout、LightGBM 选股、量化回测、超额收益、信号无效、IC 接近 0、停牌错位、股票池对比。"
agent_created: true
---

# Qlib A股因子研究：对齐陷阱与信号验证

## 铁律

1. **bin 数组按「日历位置」对齐，不是按行号。** 写数据前必须 `reindex` 到完整日历，缺口留 NaN。
2. **任何按行号做的自检都发现不了错位**（错位坐标系里它自洽）。自检必须用「日期 → 日历下标 → 读 bin」。
3. **净值不能作为「有没有 edge」的证据。** 必须跑多周期 IC + 分位组合 + 独立复现 + 随机对照。
4. **跨股票池别比超额收益**，比 Rank IC / ICIR。

---

## 坑一：bin 格式的「停牌缺口错位」（最严重，能造出 +30% 假收益）

### Qlib bin 的真实语义

```
calendars/day.txt                    每行一个交易日 YYYY-MM-DD，升序
instruments/all.txt                  每行  CODE<TAB>起始日<TAB>结束日
features/<小写code>/<field>.day.bin  前 4 字节 float32 = 起始下标 s，其后 N 个 float32
```

**第 `j` 个数值对应 `day.txt` 里的第 `(s + j)` 个交易日。位置是日历下标，不是数据行号。**

### 错误写法（看起来完全正常，实际剧毒）

```python
df = df.sort_values("date").reset_index(drop=True)
start_i = cal_idx[df["date"].iloc[0]]
vals = df["close"].to_numpy(dtype="<f4")
fp.write(struct.pack("<f", float(start_i)))
fp.write(vals.tobytes())          # ✗ 连续写入：缺口之后全部前移
```

A股停牌很常见。实测沪深300 有 **96/300 只**股票在区间中间有缺口，合计 **2076 个交易日**。一旦按行号写，缺口之后所有价格整体前移。

### 症状（记住这个组合，一看到就该怀疑错位）

| 观测 | 表现 |
|---|---|
| Rank IC | 被压到 ≈0（实测 `0.000089`）——因为横截面可比性被破坏 |
| 策略净值 | **反而虚高**（实测 `1.7567`）——错位制造了不存在的价差收益 |
| 自检 | **全部通过**——按行号比对时错位序列自洽 |
| 直接证据 | 读 bin 时 `IndexError: index N is out of bounds`，因为日历长度 > bin 长度 |

一句话：**IC≈0 但超额很漂亮，就是错位的典型指纹。**

### 正确写法

```python
df["date"] = pd.to_datetime(df["date"])
start_i = cal_idx[df["date"].iloc[0]]
end_i   = cal_idx[df["date"].iloc[-1]]
span    = cal[start_i : end_i + 1]                  # 该标的覆盖的完整日历切片

s = df.set_index("date").reindex(span)              # ★ 缺口补 NaN
vals = s[f].to_numpy(dtype="<f4")                   # ★ 保留 NaN，不要 nan_to_num
with open(fdir / f"{f}.day.bin", "wb") as fp:
    fp.write(struct.pack("<f", float(start_i)))
    fp.write(vals.tobytes())
```

**为什么必须留 NaN 而不是补 0 或前值：**

- `nan_to_num(nan=0.0)` → 价格序列出现 `100 → 0 → 100`，收益算出来是 -100%/+inf，灾难级。
- NaN 是 Qlib 的「当日无数据」语义：回测时该股当天不可成交，`TopkDropoutStrategy` 自然跳过。**免费拿到正确的停牌处理。**
- 回测日志会出现 `WARNING - qlib.online operator - $close field data contains nan`，**这是正常的**，不是错误。

### 正确的自检（按日历位置比对）

```python
cal_pos = {d: i for i, d in enumerate(cal)}
start_i = int(struct.unpack("<f", raw[:4])[0])
arr = np.frombuffer(raw[4:], dtype="<f4")
for k, dt in enumerate(pd.to_datetime(df["date"])):
    j = cal_pos[dt] - start_i                       # ★ 用日历下标定位
    assert np.isclose(arr[j], df["close"].iloc[k], rtol=1e-5, atol=1e-4)
```

顺带：**写完必须逐日校验每一只股票**，不能只抽样几只——有缺口的往往正是少数股票。

---

## 坑二：只看净值会被骗 —— 信号验证四件套

净值漂亮的原因可能有一堆：池子的风格溢价、数据错位、行情顺风、幸存者偏差。判断信号真伪要四条独立证据。

### 1. 多周期 IC（必须随持有期有结构）

```python
for h in [1, 2, 5, 10, 20]:
    fwd = close.shift(-h) / close - 1               # close 是「日期 × 股票」面板
    ics = [score.loc[dt].corr(fwd.loc[dt], method="spearman") for dt in score.index]
```

**真实信号的特征是 IC 随持有期单调上升**（低频率因子在日频上噪声大）。实测：
`1日 +0.0091 → 2日 +0.0148 → 5日 +0.0262 → 10日 +0.0367 → 20日 +0.0432`。

**日频 IC 低 ≠ 无效。** Alpha158 里真正起作用的是 `RSV60 / SUMN10 / ROC5 / STD60` 这类**中周期**因子；策略日均换手只有 0.12（平均持有约 8 天），所以决定收益的是多周期 IC。别拿日频 IC 判死刑。

### 2. 分位组合（必须单调）

按 score 分 10 组看持有期收益，Q0→Q9 应单调递增，多空要有明显价差。实测 Q0 `-0.088%` → Q9 `+0.843%`（5日），多空 `+0.93%/5日`。

### 3. 独立复现（绕开撮合引擎，最关键的一条）

用**纯 pandas** 重算组合净值，看能不能复现 Qlib 报出来的数字。

> **⚠️ 必须先对齐「换仓规则」，否则一定对不上，而且会误导你以为有 bug。**
> 我踩过这个坑：第一版复现用「每天把 topk 全部重选」（日换手 ≈ 100%），
> 而 Qlib 的 `TopkDropoutStrategy(n_drop=3)` 每天**只换 3 只**（日换手 ≈ 0.06）。
> 结果中证500 复现出 2.47~3.06，引擎只有 1.82，被误判成「落在区间外 ✗ 需排查」。
> 两者根本不是同一个策略，换手差 16 倍，收益当然差一大截。

**正确做法：忠实实现 TopkDropout 的规则**（每日卖出持仓中分数最低的 n_drop 只，
买入未持仓中分数最高的 n_drop 只），并按 Qlib 费率扣成本：

```python
def topk_dropout(sig, ret, topk, n_drop, shift_sig=0,
                 open_cost=0.0005, close_cost=0.0015):
    idx = sig.index.intersection(ret.index)
    held, recs = [], []
    for i, dt in enumerate(idx):
        j = i + shift_sig
        if j >= len(idx):
            break
        tgt = idx[j]
        s = sig.loc[dt].dropna()
        r = ret.loc[tgt]
        if len(s) < topk:
            continue
        if not held:
            held = list(s.nlargest(topk).index); turn = 1.0
        else:
            held = [c for c in held if c in s.index]                  # 剔除已无数据的
            if len(held) > topk:
                held = list(s.reindex(held).dropna().nlargest(topk).index)
            nd = min(n_drop, len(held))
            sell = list(s.reindex(held).dropna().sort_values().index[:nd]) if nd else []
            keep = [c for c in held if c not in set(sell)]
            buy = list(s.drop(labels=keep, errors="ignore").nlargest(nd).index) if nd else []
            held = keep + buy
            turn = nd / max(topk, 1)                                   # 单边换手
        vals = r.reindex(held)
        gross = float(vals.mean()) if vals.notna().any() else 0.0
        recs.append((tgt, gross - turn * (open_cost + close_cost), turn))
    out = pd.DataFrame(recs, columns=["date", "net", "turnover"]).set_index("date")
    out["cum_net"] = (1 + out["net"]).cumprod()
    return out
```

> **⚠️ 第二个必须对齐的是「涨跌停约束」，它对小中盘池影响巨大。**
> Qlib 的 `limit_threshold=0.095` 会**拒绝涨停买入 / 跌停卖出**。而动量类信号选出来的
> 股票经常当天涨停——现实里根本买不进。复现时不建模这个约束，就等于白吃了
> 「买不进去的涨停板」后面的涨幅，收益明显虚高。
>
> ```python
> dret = close.pct_change()
> tradable = (dret.abs() < limit_threshold) & close.notna() & dret.notna()
> # 买入候选要 ∩ tradable[tgt]；卖出时不可交易的持仓继续拿着（Qlib 也是这么做的）
> ```
>
> 实测沪深300：不加约束 `[1.8004, 1.9199]` → 加约束 `[1.7685, 1.8522]`。
> **池子越小盘，这个效应越强**（涨停更频繁）。

> **⚠️ 判定要单向看：复现是「上界」，不是「区间」。**
> 复现省略了限价单排队、成交额约束、停牌期间的调仓延后、逐笔成交价等摩擦，
> 所以它天然**高于**真实撮合结果。正确判定：
>
> | 情况 | 含义 |
> |---|---|
> | 引擎值 ∈ 复现区间 | ✓ 口径高度一致 |
> | 引擎值 **低于**复现上界 | ✓ 正常，说明引擎更保守（**这是常态**） |
> | 引擎值 **高于**复现上界 1% | ✗ **必须排查**——引擎凭空造出了复现拿不到的收益 |
>
> 一开始我写成双向判定（区间外就打 ✗），结果中证500 因为摩擦建模不全被误报成
> 「需排查」，白折腾一轮。**引擎偏低不是 bug，引擎偏高才是。**

实测（沪深300 全池，topk=50, n_drop=3）：

| 口径 | 净值 |
|---|---|
| Qlib 引擎报出 | **1.8611** |
| TopkDropout 复现 · 口径A（T买T+1卖） | 1.9199 |
| TopkDropout 复现 · 口径B（T+1买T+2卖＝标签口径） | 1.8004 |
| 每日全换仓（**上界参考，不与引擎对照**） | 2.1102 / 1.7013 |

**引擎值 1.8611 落在 [1.8004, 1.9199] 内 ✓**，两个时序口径把真值夹住。

**顺手得到一个免费的自洽检查**：复现出来的日单边换手 `0.062`，而 Qlib 报告 CSV 里
`turnover` 列的日均是 `0.124`——**正好 2 倍**，说明 Qlib 那列是双边口径。
两边能对上就说明换仓逻辑实现无误。

### 4. 对照组（这一步最容易漏，但最能定性）

- **等权全池**：不动模型，等权持有全池。得出「池子 + 等权」的天然基线。
- **随机信号 Topk**：多个种子跑同一回测。随机选股 ≈ 等权全池才说明**没有系统性偏差**。

实测（沪深300 全池，测试期，随机对照用**同一 TopkDropout 规则**）：
| 组合 | 净值 |
|---|---|
| 随机信号 TopkDropout（10 种子） | 1.2073（范围 1.08~1.31） |
| 等权全池 | 1.2720 |
| 模型信号（TopkDropout 独立复现） | 1.7685 ~ 1.9199 |
| Qlib 引擎 | 1.8611 |
| 沪深300 基准 | 1.1385 |

随机 ≈ 等权 → 框架无偏；模型 ≫ 随机 → 信号真实有效。

**随机对照必须跑同一换仓规则。** 用「每日全换仓」跑随机信号会得到偏低的值
（因为每天全换仓吃到的噪声更大），和等权全池对不上，反而会误导判断。

---

## 坑三：超额收益里的「等权 vs 市值加权」风格溢价

**这是最容易被误读的地方。**

实测：沪深300 全池超额年化 `34.80%`，中证500 全池超额年化 `22.25%`。但 IC 完全反过来：

| 池子 | Rank IC | ICIR | IC>0 占比 | 超额年化 |
|---|---|---|---|---|
| 沪深300 全池（300只） | 0.0035 | 0.0201 | 49.8% | **34.80%** |
| 中证500 全池（500只） | **0.0153** | **0.1122** | **54.7%** | 22.25% |

IC>0 占比 49.8% ≈ 抛硬币，说明沪深300 上模型**根本没有排序能力**，那 34.8% 从哪来？**实测等权持有沪深300 全池、不跑任何模型**就能跑出净值 1.2996 vs 基准 1.1390（**+14.10% 超额**）。大盘股里「等权」和「市值加权」差异巨大，这就是纯风格溢价。

中证500 成分股市值更接近，等权与市值加权差异小，所以它的 22.25% 才更接近真实选股能力。

**结论：**
- 跨池子比较看 **IC / ICIR**，不要看超额收益。
- 大盘股（沪深300）定价效率高，Alpha158 量价因子基本失效（IC>0 只占 49.8%）；换中盘股（中证500）IC 提升 **4.4 倍**、ICIR 提升 **5.6 倍**。

---

## 坑四：Qlib 环境与配置的常见翻车点

| 问题 | 现象 | 解法 |
|---|---|---|
| **指数被当成股票** | 指数塞进 `instruments/all.txt`，Alpha158 会把它当一只「股票」提因子、Topk 还会选它 | 指数写 `instruments/index.txt`，个股写 `all.txt`。**基准是从 `features/<code>/close.day.bin` 取数的，不依赖 instruments 列表**，所以分开不影响基准计算 |
| mlflow 3.x | `MlflowException: The filesystem tracking backend ... is in maintenance mode` | `os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")` |
| pandas 3.0 | `series[-1]` 抛 `KeyError: -1` | 一律 `.iloc` |
| 回测结束日 | `IndexError: index N is out of bounds` | 撮合要取 `calendar[i+1]`，结束日让出尾部 2~3 天 |
| 代码格式 | `INDEX_SPECS` 里 `000300` 被误判成深市 | 显式写 `SH000300`（`f"SH{code.zfill(6)}"`），别做前缀推断 |
| Windows 多进程 | `RuntimeError: An attempt has been made to start a new process before the current process has finished its bootstrapping phase` | `D.features` 会 spawn 子进程，主逻辑必须包在 `if __name__ == "__main__":` 里 |
| early stopping | `best iteration is: [2]` 只训了 2 棵树，看着像模型坏了 | 这**正常**。Alpha158 的有效因子本身就强，浅模型 ≈ 因子线性加权，反而过拟合风险低 |

### 指数/基准正确的写法

```python
for p in sorted(INDEX_DIR.glob("*.parquet")):
    data[p.stem] = load_daily(p.stem, is_index=True)   # key = "SH000905"，qlib_code 透传
    index_codes.add(p.stem)
```

---

## 坑五：akshare 取 A 股数据的多源降级

**东财 `push2his` 在机房/代理环境经常不可达**（空响应 / ProxyError），时好时坏。开跑前先探活再批量下，比逐只试错快一个数量级。

| 接口 | 源 | 备注 |
|---|---|---|
| `ak.stock_zh_a_hist()` | 东财 | 字段最全，最容易被拦 |
| `ak.stock_zh_a_daily()` | 新浪 | 稳定，成交量单位是**股** |
| `ak.stock_zh_a_hist_tx()` | 腾讯 | 稳定，单位也是**股** |
| `ak.index_stock_cons_csindex()` | 中证指数官网 | 取指数成分股；**会挂起**（见下） |
| `ak.stock_zh_index_daily()` | 新浪 | 指数日线 |

- **成交量单位**：东财是「手」（×100），新浪/腾讯是「股」。搞错 VWAP 差 100 倍。
- **akshare 没有 timeout，成分股接口会永久挂起。** 实测 `ak.index_stock_cons_csindex("000905")` 让日志 5 分钟零增长。解法：守护线程 + queue 做超时（**不要用 `ThreadPoolExecutor`，它非守护线程会在进程退出时 join 而挂住**）：
  ```python
  def _call_with_timeout(fn, timeout, *a, **kw):
      q = queue.Queue()
      t = threading.Thread(target=lambda: q.put(fn(*a, **kw)), daemon=True)
      t.start(); t.join(timeout)
      return q.get() if not q.empty() else (_ for _ in ()).throw(TimeoutError())
  ```
- **成分股名单要留档**：`data/universes/<指数代码>.txt`。既能复现，也能在网络故障时用 `--codes-file` 完全绕开该网络调用。
- **⚠️ 幸存者偏差**：`index_stock_cons_csindex` 返回的是**当前**成分股，而回测起点可能早好几年。数字只能用于**横向比较池子的相对强弱**，不能当真实现收益。

---

## 标准流程骨架

```bash
# 1. 下载（按指数取池，名单留档）
python scripts/01_download_data.py --index 000905 --size 500

# 2. 转 Qlib bin（★ 按日历位置对齐 + 全量逐日校验）
python scripts/02_to_qlib.py --universe 000905 --out qlib_data_csi500

# 3. 回测
python scripts/03_qlib_alpha158.py --qlib-dir qlib_data_csi500 \
    --benchmark SH000905 --topk 50 --tag _csi500

# 4. 信号验证（★ 必须有这一步）
python scripts/06_signal_check.py --qlib-dir qlib_data_csi500 --benchmark SH000905 --tag _csi500

# 5. 多池横向对比（★ 看 IC/ICIR，别看超额）
python scripts/05_compare_pools.py
```

### 交付检查清单

- [ ] 每只股票都逐日按日历位置校验过收盘价（不是抽样）
- [ ] 停牌缺口是 NaN，不是 0、不是前向填充
- [ ] 指数在 `instruments/index.txt`，不在 `all.txt`
- [ ] 多周期 IC 有单调结构
- [ ] 分位组合单调、多空价差明显
- [ ] 独立复现与引擎**换仓规则一致**（topk / n_drop 相同）
- [ ] 独立复现建模了**涨跌停约束**（小中盘池尤其重要）
- [ ] 引擎净值**不高于**复现上界（单向判定）
- [ ] 随机信号（同规则）≈ 等权全池
- [ ] 排名写了幸存者偏差提醒，数字标了「只可横向比较」

---

## 单标的短期预测：四个必查陷阱

做「某只股票未来几天涨跌概率」时，直接 bootstrap / 条件统计会给出**看起来精确、
实际有偏**的数字。上结论前必须查这四条（实测数据来自港股 02513，175 个交易日）。

### 1. 漂移在短样本上不可识别 —— 所以别去「估」它，直接设为零

**(a) 定量论据**（Merton 1980 附录 A；Ostrov & Das 2023, SSRN 4662525）

样本均值的标准误是 **σ/√T，其中 T 是年数，不是观测个数**。提高采样频率
（日频→分钟频）对这个标准误毫无帮助 —— 这是最反直觉的一点。

实测（港股 02513，174 个交易日）：

| 量 | 值 |
|---|---|
| 年化 σ | 185.9% |
| 样本长度 T | 0.690 年 |
| 漂移标准误 σ/√T | **223.7%** |
| 观测年化漂移 | +394.7%（t = 1.76，无法拒绝零） |
| 漂移的 95% 置信区间 | **[−44%, +833%]** |
| 把 SE 压到 10% 需要 | **346 年** |

结论：短样本上「漂移」与「零」统计上不可区分。**任何「扣多少趋势 / 用多长窗口」的
选择都不是在挑更准的估计，而是在挑噪声的一个具体实现。**

**样本长度怎么改变结论 —— 同一套方法的两次实测对照**：

| 项 | 港股 02513（次新股） | 美股 MU（成熟股） |
|---|---|---|
| 样本 | 174 天（0.69 年） | 3907 天（15.5 年） |
| 年化 σ | 185.9% | 59.1% |
| 漂移标准误 σ/√T | 223.7% | **15.0%** |
| t 值 | 1.76（不显著） | **3.29（显著）** |
| 漂移 95% CI | [−44%, +833%] | **[+20%, +79%]** |
| 「单日跌≤-10%」独立波段 | 3（不可用） | **17（可用）** |

**动笔写结论前，先看样本长度，再决定「漂移」这个话题能不能谈。**
σ/√T 会把这件事量化：SE > |漂移估计| 就别谈漂移。

另外，漂移**显著为正**也不等于该外推到未来几天：
实测 MU 5 日漂移贡献仅 +0.98%，而 5 日波动 8.3% —— 短期里波动压过漂移一个量级。

**(b) 四种口径实测对比**（未来 5 日）

| 口径 | 上涨概率 | 5% 分位 | 95% 分位 |
|---|---|---|---|
| 全历史 bootstrap | 55.9%（有偏） | −29.4% | +61.0% |
| 去漂移 bootstrap | 42.3% | −34.9% | +48.5% |
| 近 60 日窗口 bootstrap | **34.1%** | −38.7% | **+32.9%** |
| **FHS-GARCH（推荐）** | **45.0%** | **−31.0%** | **+46.9%** |

近 60 日窗口是最糟的：窗口自带近期趋势，而「选 60」这个动作本身就是任意的。
看它的 95% 分位只有 +32.9% —— 上行空间被「最近一直在跌」人为压扁了。
（Pesaran & Timmermann 2007, *J. Econometrics* 137(1) 研究这个 bias-variance 权衡：
短窗偏差小但方差大、长窗反之；他们的建议是**别选单一窗口，而是组合多个窗口**，
而不是换一个更短的。）

**(c) 正确做法：FHS（Filtered Historical Simulation）**

Barone-Adesi, Bourgoin & Giannopoulos (1998), *Risk* 11(8), 100–103（Hull & White 1998 独立提出）。四步：

1. 用 **GJR-GARCH(1,1)** 拟合，把收益拆成「条件波动率 × 近似 i.i.d. 的标准化残差」。
   选 GJR 是因为股票有**杠杆效应**（跌时波动涨得更快）—— 这正是「最近一直在跌」
   这句话在统计上的真实含义：**变的不是漂移，是波动率**。
2. 对标准化残差 z 做 bootstrap（保留肥尾与偏度）。
3. 用 GARCH **递归**生成未来每天的波动率，把 z 乘回去。递归是关键 —— 它让模拟路径
   自带波动聚集，而不是套用一个固定日波动率（那正是朴素 bootstrap 的毛病）。
4. 漂移**显式设为 0**（依据 (a)），不确定性全部由波动率通道表达。

实测（ω=37.26 α=0.121 γ=0.039 β=0.587，持续性 0.727）：当前条件年化波动率
**160.7% < 无条件 186.4%**，说明波动正在收敛 —— 这是朴素 bootstrap 完全看不出的信息。

```python
from arch import arch_model
res = arch_model(r*100, mean="Constant", vol="GARCH", p=1, o=1, q=1,
                 dist="skewt").fit(disp="off")
cv = np.asarray(res.conditional_volatility).ravel()   # arch 8.x 返回 ndarray，不是 Series
z  = (r*100 - res.params["mu"]) / cv                  # 标准化残差，已剥掉波动聚集
# 递归：s2 = ω + α·e² + γ·1(e<0)·e² + β·s2；e = sqrt(s2)·z_k；当期收益 = e（漂移 0）
```

**坑**：arch 8.x 起 `conditional_volatility` 返回 **ndarray**（旧版是 Series），
写 `.iloc[-1]` 会抛 `AttributeError: 'numpy.ndarray' object has no attribute 'iloc'`。
另外 `last_sigma_ann` 若存的是「百分数数值」（160.7），打印要用 `{:.1f}%` 而不是
`{:.1%}`，否则会多乘 100（曾把 160.7% 显示成 16068.7%）。

### 2. 条件形态的事件不独立（最容易被骗的一处）

「历史上大跌之后会反弹」这类统计，风险在于**事件扎堆**：同一个暴跌波段里连续触发
10 次信号，看着是 10 个样本，实际只是 1 个样本被数了 10 遍。

实测：`单日跌幅 ≤ -10%` 触发 15 次，但相邻间隔中位数只有 11 个交易日 → 独立波段仅 **3** 个；
`单日跌幅 ≤ -5%` 触发 19 次 → 独立波段仅 **1** 个。

**做法**：去重（间隔 >3 日）→ 按间隔 >20 日切波段 → **报波段数而不是事件数**；
波段数 < 8 就直接标注「不具备统计意义」。同时把触发日在报告里列出来，让人一眼看出扎堆。

### 3. 别把「方向」当可预测的，只报「幅度」

短期方向基本不可提取信息（上涨概率贴着 50% 抖动）。稳定可估的是波动幅度：
未来 N 日的 5%/25%/50%/75%/95% 分位价格。**这个宽度本身就是最有用的输出**
—— 它不是预测，是风险计量。

### 4. 历史统计盖不住「新信息」

实测案例：该股当日 -12.40% 的直接原因是海外两家 AI 实验室同日发布低价模型
（成本降 40%~50%），属于**商业模式层面的重新定价**。历史上从未发生同类事件，
任何 K 线统计对它都是零信息量。

**结论里必须显式写出这一点**，否则等于假装历史能解释未来。这也正是「先看消息面、
再看统计面」的顺序不能反的原因。

### 输出模板（照这个结构写结论）

1. 样本量够不够（不足 250 日就先声明所有数字的置信度都低）
2. 现在在哪（回撤、均线乖离、RSI、波动率分位）
3. 能估的是什么（未来 N 日分位区间，明确说是幅度不是方向）
4. 历史上类似形态（**带波段数**，样本少就直接说没意义）
5. 这次有什么历史没涵盖的新信息
6. 不构成投资建议
