# -*- coding: utf-8 -*-
"""三个可交易模式能不能赚钱：扣费 + 无前视 + 分段报告

模式：
  A 市场因子动量：昨日市场因子(PC1)涨 -> 今日持高载荷篮子；跌 -> 空仓
  B 横截面动量  ：过去 7/30 日涨幅前 20 -> 今日持有
  B2 横截面反转 ：过去 3 日跌幅前 20 -> 今日持有
  C 簇内传导    ：某簇昨日有人领涨(>5%) -> 今日买该簇滞后的 3 个
基准：
  D 等权持有全部币（buy&hold）
  E BTC 持有
费用：单边 0.1%（现货 taker 量级）；另给 0.05% 对照
无前视：PC1 载荷与簇标签每 60 天，用过去 120 天滚动重估
输出：evidence/backtest_modes.txt / .png
"""
import bisect
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data")
EV = os.path.join(ROOT, "evidence")
os.makedirs(EV, exist_ok=True)

MIN_BARS = 800
LOOKBACK = 120
REFIT = 60
FEE_MAIN = 0.001
FEE_ALT = 0.0005
ANN = 365

lines = []


def say(s=""):
    print(s, flush=True)
    lines.append(str(s))


# ---------- 数据 ----------
df = pd.read_csv(os.path.join(DATA, "klines_1d_top502.csv"))
df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_localize(None)
cnt = df.groupby("symbol").size()
keep = cnt[cnt >= MIN_BARS].index
px = (df[df["symbol"].isin(keep)]
      .pivot_table(index="open_time", columns="symbol", values="close")
      .sort_index().dropna())
ret = px.pct_change().iloc[1:]
dates = ret.index
cols = list(ret.columns)
n = len(dates)
say("对齐: %d 天 x %d 币  %s .. %s" % (n, len(cols), dates[0].date(), dates[-1].date()))
say("费用假设: 单边 %.2f%%（往返 %.2f%%），另给 %.2f%% 对照"
    % (FEE_MAIN * 100, 2 * FEE_MAIN * 100, FEE_ALT * 100))

# ---------- 滚动重估（无前视）----------
refits = []
for i in range(LOOKBACK, n, REFIT):
    win = ret.iloc[i - LOOKBACK:i]
    Z = win.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="any")
    if Z.shape[1] < 20:
        continue
    Z = (Z - Z.mean()) / Z.std().replace(0, np.nan)
    Z = Z.dropna(axis=1, how="any")
    C = np.corrcoef(Z.values, rowvar=False)
    w, v = np.linalg.eigh(C)
    order = np.argsort(w)[::-1]
    pc1 = pd.Series(v[:, order[0]], index=list(Z.columns))
    D = np.clip(1.0 - C, 0.0, None)
    np.fill_diagonal(D, 0.0)
    Zl = linkage(squareform(D, checks=False), method="average")
    lab = fcluster(Zl, t=0.5, criterion="distance")
    refits.append((i, pc1, pd.Series(lab, index=list(Z.columns))))
_starts = [r[0] for r in refits]
say("滚动重估: %d 次（窗口 %d 天 / 步长 %d 天）" % (len(refits), LOOKBACK, REFIT))


def get_ref(i):
    """用第 i-1 天之前已完成的估计"""
    k = bisect.bisect_right(_starts, i - 1) - 1
    k = max(k, 0)
    return refits[k][1], refits[k][2]


# ---------- 信号 ----------
def w_eq(syms):
    syms = list(syms)
    if not syms:
        return pd.Series(dtype=float)
    return pd.Series(1.0 / len(syms), index=syms)


def sig_A(i, topn=30):
    pc1, _ = get_ref(i)
    r_prev = ret.iloc[i - 1]
    common = [s for s in pc1.index if s in r_prev.index]
    if len(common) < 10:
        return pd.Series(dtype=float)
    score = float((r_prev[common] * pc1[common]).sum())
    if score <= 0:
        return pd.Series(dtype=float)
    top = pc1.abs().sort_values(ascending=False).head(topn).index
    return w_eq([s for s in top if s in r_prev.index])


def sig_B(i, lb=7, k=20, ascend=False):
    past = ret.iloc[max(0, i - lb):i]
    if len(past) < lb:
        return pd.Series(dtype=float)
    cum = (1.0 + past).prod() - 1.0
    cum = cum.dropna()
    if len(cum) < k:
        return pd.Series(dtype=float)
    sel = cum.sort_values(ascending=ascend).head(k).index
    return w_eq(sel)


def sig_C(i, thr=0.05, k=3):
    _, lab = get_ref(i)
    r_prev = ret.iloc[i - 1]
    picks = []
    for cid, grp in lab.groupby(lab):
        if len(grp) < 3:
            continue
        syms = [s for s in grp.index if s in r_prev.index]
        rr = r_prev[syms].dropna()
        if len(rr) < 3 or rr.max() < thr:
            continue
        laggards = rr.sort_values().head(k)
        picks += list(laggards.index)
    return w_eq(picks)


# ---------- 回测引擎 ----------
def run(sig_fn, fee, start=LOOKBACK + 1):
    nav = 1.0
    prev = pd.Series(dtype=float)
    curve, turns, fees = [], [], []
    for i in range(start, n):
        w = sig_fn(i)
        r = ret.iloc[i]
        idx = w.index.union(prev.index)
        dw = w.reindex(idx).fillna(0.0) - prev.reindex(idx).fillna(0.0)
        turn = float(dw.abs().sum())
        cost = turn * fee
        gross = float((w.reindex(r.index).fillna(0.0) * r.fillna(0.0)).sum())
        nav *= (1.0 + gross - cost)
        curve.append(nav)
        turns.append(turn)
        fees.append(cost)
        prev = w
    s = pd.Series(curve, index=dates[start:])
    return s, np.array(turns), np.array(fees)


def stats(s, turns, fees):
    r = s.pct_change().dropna()
    if len(r) == 0:
        return {}
    yrs = len(r) / ANN
    total = float(s.iloc[-1] - 1.0)
    cagr = float(s.iloc[-1] ** (1.0 / yrs) - 1.0) if s.iloc[-1] > 0 else -1.0
    vol = float(r.std() * np.sqrt(ANN))
    sharpe = float(r.mean() / r.std() * np.sqrt(ANN)) if r.std() > 0 else 0.0
    dd = float((s / s.cummax() - 1.0).min())
    return dict(total=total, cagr=cagr, vol=vol, sharpe=sharpe, mdd=dd,
                win=float((r > 0).mean()), turnover=float(np.mean(turns)),
                fee_drag=float(np.sum(fees)))


# 基准
def bench_hold(syms, fee):
    w0 = w_eq(syms)
    nav = 1.0
    prev = pd.Series(dtype=float)
    curve, turns, fees = [], [], []
    start = LOOKBACK + 1
    for i in range(start, n):
        r = ret.iloc[i]
        idx = w0.index.union(prev.index)
        dw = w0.reindex(idx).fillna(0.0) - prev.reindex(idx).fillna(0.0)
        turn = float(dw.abs().sum())
        cost = turn * fee
        gross = float((w0.reindex(r.index).fillna(0.0) * r.fillna(0.0)).sum())
        nav *= (1.0 + gross - cost)
        curve.append(nav)
        turns.append(turn)
        fees.append(cost)
        prev = w0
    return pd.Series(curve, index=dates[start:]), np.array(turns), np.array(fees)


MODES = [
    ("A 市场因子动量", lambda i: sig_A(i)),
    ("B 横截面动量7d", lambda i: sig_B(i, 7, 20)),
    ("B 横截面动量30d", lambda i: sig_B(i, 30, 20)),
    ("B2 横截面反转3d", lambda i: sig_B(i, 3, 20, ascend=True)),
    ("C 簇内传导", lambda i: sig_C(i)),
]

say("")
say("=== 扣费后表现（单边 %.2f%%，%s..%s）===" % (FEE_MAIN * 100, dates[LOOKBACK + 1].date(), dates[-1].date()))
hdr = "%-18s %9s %9s %7s %7s %7s %8s %8s" % ("模式", "总收益", "年化", "波动", "夏普", "最大回撤", "日均换手", "费用拖累")
say(hdr)
say("-" * len(hdr))

curves = {}
for name, fn in MODES:
    s, t, f = run(fn, FEE_MAIN)
    st = stats(s, t, f)
    curves[name] = s
    say("%-18s %8.1f%% %8.1f%% %6.1f%% %7.2f %7.1f%% %7.2f %7.1f%%"
        % (name, st["total"] * 100, st["cagr"] * 100, st["vol"] * 100,
           st["sharpe"], st["mdd"] * 100, st["turnover"], st["fee_drag"] * 100))

for name, syms in [("D 等权持有全部", cols), ("E BTC 持有", ["BTCUSDT"])]:
    s, t, f = bench_hold(syms, FEE_MAIN)
    st = stats(s, t, f)
    curves[name] = s
    say("%-18s %8.1f%% %8.1f%% %6.1f%% %7.2f %7.1f%% %7.2f %7.1f%%"
        % (name, st["total"] * 100, st["cagr"] * 100, st["vol"] * 100,
           st["sharpe"], st["mdd"] * 100, st["turnover"], st["fee_drag"] * 100))

# 零费用对照：看费用是不是决定性的
say("")
say("=== 零费用（理想化）对照——用来分离「信号有没有用」和「费用吃不吃得掉」===")
say(hdr)
say("-" * len(hdr))
for name, fn in MODES:
    s, t, f = run(fn, 0.0)
    st = stats(s, t, f)
    say("%-18s %8.1f%% %8.1f%% %6.1f%% %7.2f %7.1f%% %7.2f %7.1f%%"
        % (name, st["total"] * 100, st["cagr"] * 100, st["vol"] * 100,
           st["sharpe"], st["mdd"] * 100, st["turnover"], st["fee_drag"] * 100))

# 前后半段（伪 walk-forward 稳定性）
say("")
say("=== 分段稳定性（扣费后年化，前半 vs 后半）===")
half = (n - (LOOKBACK + 1)) // 2
for name, fn in MODES:
    s, t, f = run(fn, FEE_MAIN)
    r = s.pct_change().dropna()
    h1, h2 = r.iloc[:half], r.iloc[half:]
    a1 = float((1 + h1).prod() ** (ANN / len(h1)) - 1) * 100
    a2 = float((1 + h2).prod() ** (ANN / len(h2)) - 1) * 100
    say("%-18s  前半 %+7.1f%%   后半 %+7.1f%%" % (name, a1, a2))

# ---------- 图 ----------
fig, axes = plt.subplots(2, 1, figsize=(13, 9))
ax = axes[0]
for name, s in curves.items():
    ax.plot(s.index, s.values, lw=1.2, label=name)
ax.set_yscale("log")
ax.set_title("Net value after fees (single-side %.2f%%)" % (FEE_MAIN * 100), fontsize=11)
ax.set_ylabel("NAV (log)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1]
for name, fn in MODES:
    s0, _, _ = run(fn, 0.0)
    s1, _, _ = run(fn, FEE_MAIN)
    ax.plot(s1.index, s1.values / s0.values, lw=1.2, label=name)
ax.axhline(1.0, color="0.7", lw=0.8)
ax.set_title("Fee impact ratio: NAV(after fees) / NAV(no fees)", fontsize=11)
ax.set_ylabel("ratio"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout()
p1 = os.path.join(EV, "backtest_modes.png")
plt.savefig(p1, dpi=125)
plt.close()

say("")
say("saved: %s" % p1)
with open(os.path.join(EV, "backtest_modes.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
