# -*- coding: utf-8 -*-
"""多空中性版：把「市场涨跌」与「信号有没有 alpha」分开

背景：2024-11~2026-10 样本期 BTC -17.5%、等权全币 -71.4%（熊市），
      纯多头必亏，无法判断信号本身有没有用。
做法：同一信号同时做多 top 篮子 / 做空 bottom 篮子（各 50% 本金、美元中性），
      扣双边手续费。
注意：现货不能做空，空腿需永续合约，资金费率未计入（标注）。
输出：evidence/backtest_ls.txt / .png
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

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data")
EV = os.path.join(ROOT, "evidence")
os.makedirs(EV, exist_ok=True)

MIN_BARS, LOOKBACK, REFIT = 800, 120, 60
FEE = 0.001
ANN = 365
lines = []


def say(s=""):
    print(s, flush=True)
    lines.append(str(s))


df = pd.read_csv(os.path.join(DATA, "klines_1d_top502.csv"))
df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_localize(None)
cnt = df.groupby("symbol").size()
keep = cnt[cnt >= MIN_BARS].index
px = (df[df["symbol"].isin(keep)]
      .pivot_table(index="open_time", columns="symbol", values="close")
      .sort_index().dropna())
ret = px.pct_change().iloc[1:]
dates, cols, n = ret.index, list(ret.columns), len(ret)
say("样本: %d 天 x %d 币  %s .. %s" % (n, len(cols), dates[0].date(), dates[-1].date()))

# 样本期市场状态——决定「多头必亏」不是信号问题
seg = ret.iloc[LOOKBACK + 1:]
say("样本期（回测窗口）市场状态：等权全币累计 %+.1f%%，BTC %+.1f%%"
    % (((1 + seg.mean(axis=1)).prod() - 1) * 100, ((1 + seg["BTCUSDT"]).prod() - 1) * 100))

refits = []
for i in range(LOOKBACK, n, REFIT):
    win = ret.iloc[i - LOOKBACK:i].replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="any")
    if win.shape[1] < 20:
        continue
    Z = ((win - win.mean()) / win.std().replace(0, np.nan)).dropna(axis=1, how="any")
    C = np.corrcoef(Z.values, rowvar=False)
    w, v = np.linalg.eigh(C)
    order = np.argsort(w)[::-1]
    PC1 = pd.Series(v[:, order[0]], index=list(Z.columns))
    D = np.clip(1.0 - C, 0.0, None)
    np.fill_diagonal(D, 0.0)
    lab = fcluster(linkage(squareform(D, checks=False), method="average"), t=0.5, criterion="distance")
    refits.append((i, PC1, pd.Series(lab, index=list(Z.columns))))
_starts = [r[0] for r in refits]


def get_ref(i):
    k = max(bisect.bisect_right(_starts, i - 1) - 1, 0)
    return refits[k][1], refits[k][2]


# ---- 信号：返回 (long列表, short列表) ----
def sel_mom(i, lb, k=20):
    past = ret.iloc[max(0, i - lb):i]
    if len(past) < lb:
        return [], []
    cum = ((1.0 + past).prod() - 1.0).dropna()
    o = cum.sort_values(ascending=False)
    return list(o.index[:k]), list(o.index[-k:])


def sel_cluster(i, thr=0.05, k=3):
    _, lab = get_ref(i)
    r_prev = ret.iloc[i - 1]
    lo, sh = [], []
    for _, grp in lab.groupby(lab):
        if len(grp) < 3:
            continue
        syms = [s for s in grp.index if s in r_prev.index]
        rr = r_prev[syms].dropna()
        if len(rr) < 3 or rr.max() < thr:
            continue
        o = rr.sort_values()
        lo += list(o.index[:k])
        sh += list(o.index[-k:])
    return lo, sh


def sel_factor(i, topn=30):
    pc1, _ = get_ref(i)
    a = pc1.abs().sort_values(ascending=False)
    return list(a.index[:topn]), list(a.index[-topn:])


def run(sig_fn, ls=False, fee=FEE, start=LOOKBACK + 1):
    """sig_fn(i) -> (long列表, short列表)；ls=False 纯多头满仓，True 多空各 50% 美元中性"""
    nav, prev, c = 1.0, pd.Series(dtype=float), []
    for i in range(start, n):
        lo, sh = sig_fn(i)
        w = pd.Series(dtype=float)
        if lo:
            w = pd.concat([w, pd.Series(0.5 / len(lo) if ls else 1.0 / len(lo), index=lo)])
        if ls and sh:
            w = pd.concat([w, pd.Series(-0.5 / len(sh), index=sh)])
        w = w.groupby(level=0).sum()
        r = ret.iloc[i]
        idx = w.index.union(prev.index)
        turn = float((w.reindex(idx).fillna(0) - prev.reindex(idx).fillna(0)).abs().sum())
        gross = float((w.reindex(r.index).fillna(0.0) * r.fillna(0.0)).sum())
        nav *= (1.0 + gross - turn * fee)
        c.append(nav)
        prev = w
    return pd.Series(c, index=dates[start:])


def stat(s):
    r = s.pct_change().dropna()
    yrs = len(r) / ANN
    return dict(tot=float(s.iloc[-1] - 1), cagr=float(s.iloc[-1] ** (1 / yrs) - 1),
                vol=float(r.std() * np.sqrt(ANN)),
                sh=float(r.mean() / r.std() * np.sqrt(ANN)) if r.std() > 0 else 0.0,
                mdd=float((s / s.cummax() - 1).min()), win=float((r > 0).mean()))


MODES = [
    ("动量7d", lambda i: sel_mom(i, 7)),
    ("动量30d", lambda i: sel_mom(i, 30)),
    ("簇内传导", lambda i: sel_cluster(i)),
    ("因子暴露(高/低beta)", lambda i: sel_factor(i)),
]

say("")
say("=== 纯多头 vs 多空中性（扣费单边 %.2f%%）===" % (FEE * 100))
hdr = "%-20s %10s %10s %8s %8s %8s" % ("模式", "多头总收益", "多空总收益", "多头夏普", "多空夏普", "多空回撤")
say(hdr)
say("-" * len(hdr))

curves = {}
for name, fn in MODES:
    sl = run(fn, False)
    ss = run(fn, True)
    a, b = stat(sl), stat(ss)
    curves[name + " 多头"] = sl
    curves[name + " 多空"] = ss
    say("%-20s %9.1f%% %9.1f%% %8.2f %8.2f %7.1f%%"
        % (name, a["tot"] * 100, b["tot"] * 100, a["sh"], b["sh"], b["mdd"] * 100))

# 基准
b_ew = run(lambda i: (cols, []), False)
b_btc = run(lambda i: (["BTCUSDT"], []), False)
curves["等权全币"] = b_ew
curves["BTC"] = b_btc
say("%-20s %9.1f%% %9s %8.2f" % ("等权全币(基准)", stat(b_ew)["tot"] * 100, "-", stat(b_ew)["sh"]))
say("%-20s %9.1f%% %9s %8.2f" % ("BTC(基准)", stat(b_btc)["tot"] * 100, "-", stat(b_btc)["sh"]))

# 多空的分段稳定性
say("")
say("=== 多空组合分段稳定性（前半 / 后半 年化）===")
half = (n - (LOOKBACK + 1)) // 2
for name, fn in MODES:
    ss = run(fn, True)
    r = ss.pct_change().dropna()
    h1, h2 = r.iloc[:half], r.iloc[half:]
    say("%-22s 前半 %+7.1f%%   后半 %+7.1f%%"
        % (name, ((1 + h1).prod() ** (ANN / len(h1)) - 1) * 100,
           ((1 + h2).prod() ** (ANN / len(h2)) - 1) * 100))

fig, ax = plt.subplots(figsize=(13, 7))
for name, s in curves.items():
    ax.plot(s.index, s.values, lw=1.3, label=name)
ax.set_yscale("log")
ax.set_title("纯多头 vs 多空中性（扣费 单边0.10%%）  样本 %s..%s"
             % (dates[LOOKBACK + 1].date(), dates[-1].date()), fontsize=11)
ax.set_ylabel("NAV (log)")
ax.legend(fontsize=8, ncol=2)
ax.grid(alpha=0.3)
plt.tight_layout()
p1 = os.path.join(EV, "backtest_ls.png")
plt.savefig(p1, dpi=125)
plt.close()
say("")
say("saved: %s" % p1)
with open(os.path.join(EV, "backtest_ls.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
