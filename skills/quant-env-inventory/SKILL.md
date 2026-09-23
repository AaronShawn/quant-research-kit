---
name: quant-env-inventory
description: 盘点本机量化环境与工具，产出「哪个库装在哪个解释器里」的字典供后续调用。当需要知道「某个量化库装在哪个 Python 环境」「venv 在哪」「数据源能不能通」「该用哪个脚本跑」时使用；环境变动后也用它刷新清单。触发词：环境在哪、装在哪、用什么跑、哪个环境、有哪些量化工具、能不能用 X 框架、环境清单、盘点环境、资产清单。
agent_created: true
---

# 量化环境盘点

## 为什么要专门做这件事

一台机器上通常有多个**互不相通**的 Python 环境：系统 Python、conda 环境、项目 venv、
工具自带的私有 venv。于是「某个包到底装在哪」和「该用哪个解释器跑」这两个问题，
每次重新探测要花十几分钟，而且探法不对还会得到**错误答案**——
比如 `python -c "import vnpy"` 用的是 PATH 里那个解释器，不是你心里想的那个。

本技能定义盘点的标准做法和清单结构，产出物是一份可直接查询的字典。

---

## 一、盘点四步

### 第 1 步：找出所有解释器

不要只信 `which python`。**`pyvenv.cfg` 是虚拟环境的标志文件**，比找 `python.exe` 更快更准：

```bash
# Linux / macOS
find ~ -maxdepth 5 -name "pyvenv.cfg" 2>/dev/null
ls -d /usr/bin/python3* /usr/local/bin/python3* \
      ~/.pyenv/versions/*/bin/python* \
      ~/miniconda3/envs/*/bin/python* ~/anaconda3/envs/*/bin/python* 2>/dev/null

# Windows (Git Bash)
find /c/Users/<你> -maxdepth 6 -iname "python.exe" -not -path "*/node_modules/*" 2>/dev/null
```

### 第 2 步：对每个环境列包

**必须用绝对路径调用**，不要依赖 PATH：

```bash
"<环境路径>/Scripts/python.exe" -m pip list    # Windows
"<环境路径>/bin/python" -m pip list             # Linux / macOS
```

### 第 3 步：按领域归类（不要只是罗列）

裸的 `pip list` 有几百行，没法查。按「做什么」分类才有价值：

| 类别 | 关键词 |
|---|---|
| 回测框架 | vnpy, backtrader, zipline, rqalpha, pyalgotrade, wonder-trader |
| 因子 / 机器学习 | qlib, lightgbm, xgboost, catboost, alphalens, scikit-learn |
| 数据获取 | akshare, tushare, baostock, yfinance, ccxt, futu, jqdatasdk, efinance |
| 风险 / 波动率 | arch, statsmodels, empyrical, quantstats, pyfolio |
| 组合优化 | cvxpy, riskfolio-lib, PyPortfolioOpt |
| 技术指标 | TA-Lib, pandas-ta, talib |
| 交互研究 | jupyter, jupyterlab, notebook |
| 可视化 | matplotlib, plotly, mplfinance, seaborn |

### 第 4 步：探测数据源可用性（最容易漏的一步）

**装得上 ≠ 连得通。** 网络环境、代理、地域封锁会让某个数据源**整体失效**，
而报错信息（`ProxyError` / `Timeout` / `ConnectionError`）常被误读成「某只股票取不到」，
从而在错误的源上反复重试。

逐个源实际请求一次并记录结果：

```python
import akshare as ak
sources = {
    "新浪-港股": lambda: ak.stock_hk_daily(symbol="02513", adjust=""),
    "新浪-美股": lambda: ak.stock_us_daily(symbol="MU", adjust="qfq"),
    "东财-A股":  lambda: ak.stock_zh_a_hist(symbol="600519", period="daily",
                              start_date="20260101", end_date="20260923"),
}
for name, fn in sources.items():
    try:
        print(f"[OK  ] {name}  {fn().shape}")
    except Exception as e:
        print(f"[FAIL] {name}  {type(e).__name__}: {str(e)[:80]}")
```

**务必自己加超时**：akshare 部分接口**没有 timeout，会永久挂起**
（实测 `ak.index_stock_cons_csindex` 取指数成分股就会卡死）。
用守护线程 + queue 包一层，不要用 `ThreadPoolExecutor`——
它的非守护线程会在进程退出时 join，照样挂住。

---

## 二、清单模板（把你盘出来的结果填进去）

```markdown
## Python 环境

| 环境 | 路径 | 用途 | 包数 |
|---|---|---|---|
| 项目 venv | <绝对路径> | 量化主力 | 234 |
| 系统 Python | <绝对路径> | 通用 / 另一套框架 | 88 |

## 库分布（★ = 只存在于该环境）

| 库 | 环境A | 环境B | 用途 |
|---|:---:|:---:|---|
| vnpy | ✅ 4.4.0 ★ | ❌ | 事件驱动回测 / 实盘 |
| backtrader | ❌ | ✅ 1.9.78 ★ | 另一套回测框架 |

## 数据源可用性（实测）

| 源 | 状态 | 备注 |
|---|:---:|---|
| 新浪 | ✅ | 主力源 |
| 东财 | ❌ | 代理阻断，全部接口不可用 |

## 脚本 ↔ 用途

| 我想做…… | 用这个 |
|---|---|
| 单标的全面分析 | `08_stock_probe.py` |
| 因子回测 | `01 → 02 → 03` |
```

**每次盘点后把日期写在清单顶部**，便于判断新鲜度。

---

## 三、七个高频坑

1. **`ModuleNotFoundError` 的第一嫌疑是「包在另一个环境里」**，而不是「没装」。
   先 `pip list` 查别的解释器，再决定装不装。
2. **`pip list` 查的不是你以为的那个环境** —— 永远用绝对路径调用解释器。
3. **装得上但连不通** —— 数据源必须实测（第 4 步），别信"装好了就能用"。
4. **venv 不能靠移动来迁移** —— 路径硬编码在 `Scripts/*.exe`、`pyvenv.cfg`、
   `*.dist-info` 里，移动即损坏。要迁移就 `pip freeze > req.txt` 再重建。
5. **同名库不同版本可能完全不同** —— 记**版本号**，不只记包名。
   例：`vnpy_riskmanager` 2.0.0 只有 cp313 的 wheel，在 Python 3.11 上会尝试源码编译，
   失败后**回滚整个安装事务**，把前面装好的包一起卸掉。
6. **`pip freeze` 会把无关依赖一起带上** —— 用 `pip list --not-required` 看顶层依赖树，
   或手工筛，别直接把 freeze 结果当交付清单。
7. **沙箱 / 受限环境里的额外限制** —— 例如批量文件删除可能被安全守卫拦截并杀进程
   （表现为「脚本跑到一半无声退出」）。这类环境的盘点要额外记录「哪些操作会触发限制」。

---

## 四、何时刷新

- 装新包、升级 / 降级、换 Python 版本
- 换机器、重装系统
- 出现「明明装了却 import 不到」
- 新增数据源，或发现某个源失效
- 距上次盘点超过一个月

---

## 五、与其它技能的关系

| 技能 | 职责 |
|---|---|
| `quant-full-analysis` | 拿到用户需求后跑九层分析流程 |
| `qlib-ashare-factor-research` | A股股票池的因子研究流水线 |
| **`quant-env-inventory`** | **本文件**：回答「用什么跑、装在哪」 |

分析类技能在动手前应先查本清单，避免用错解释器或用错数据源。
