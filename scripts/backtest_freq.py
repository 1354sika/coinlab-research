# -*- coding: utf-8 -*-
"""换手率是不是杀手：同一信号，调仓周期 1/7/14/30 天对比

动机：日频换手 1.7/天 x 0.1% 单边 = 年化费用 >100%，任何信号都会被吃光。
      本脚本把「信号好坏」与「交易成本」分离：信号相同，只改调仓频率。
输出：evidence/backtest_freq.txt / .png
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
MIN_BARS, LOOKBACK, REFIT, FEE, ANN = 800, 120, 60, 0.001, 365
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
start = LOOKBACK + 1

refits = []
for i in range(LOOKBACK, n, REFIT):
    win = ret.iloc[i - LOOKBACK:i].replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="any")
    if win.shape[1] < 20:
        continue
    Z = ((win - win.mean()) / win.std().replace(0, np.nan)).dropna(axis=1, how="any")
    C = np.corrcoef(Z.values, rowvar=False)
    w_, v = np.linalg.eigh(C)
    PC1 = pd.Series(v[:, np.argsort(w_)[::-1][0]], index=list(Z.columns))
    D = np.clip(1.0 - C, 0.0, None)
    np.fill_diagonal(D, 0.0)
    lab = fcluster(linkage(squareform(D, checks=False), method="average"), t=0.5, criterion="distance")
    refits.append((i, PC1, pd.Series(lab, index=list(Z.columns))))
_starts = [r[0] for r in refits]


def get_ref(i):
    k = max(bisect.bisect_right(_starts, i - 1) - 1, 0)
    return refits[k][1], refits[k][2]


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


def sel_mom(i, lb=7, k=20):
    past = ret.iloc[max(0, i - lb):i]
    if len(past) < lb:
        return [], []
    cum = ((1.0 + past).prod() - 1.0).dropna().sort_values(ascending=False)
    return list(cum.index[:k]), list(cum.index[-k:])


def build(lo, sh, ls):
    w = pd.Series(dtype=float)
    if lo:
        w = pd.concat([w, pd.Series(0.5 / len(lo) if ls else 1.0 / len(lo), index=lo)])
    if ls and sh:
        w = pd.concat([w, pd.Series(-0.5 / len(sh), index=sh)])
    return w.groupby(level=0).sum()


def run(sig_fn, ls, hold, fee):
    nav, prev, c, turns = 1.0, pd.Series(dtype=float), [], []
    w = pd.Series(dtype=float)
    for i in range(start, n):
        if (i - start) % hold == 0:
            lo, sh = sig_fn(i)
            w = build(lo, sh, ls)
        r = ret.iloc[i]
        idx = w.index.union(prev.index)
        t = float((w.reindex(idx).fillna(0.0) - prev.reindex(idx).fillna(0.0)).abs().sum())
        gross = float((w.reindex(r.index).fillna(0.0) * r.fillna(0.0)).sum())
        nav *= (1.0 + gross - t * fee)
        c.append(nav)
        turns.append(t)
        prev = w
    return pd.Series(c, index=dates[start:]), float(np.mean(turns) * ANN / hold)


def stat(s):
    r = s.pct_change().dropna()
    yrs = len(r) / ANN
    return dict(tot=float(s.iloc[-1] - 1), cagr=float(s.iloc[-1] ** (1 / yrs) - 1),
                sh=float(r.mean() / r.std() * np.sqrt(ANN)) if r.std() > 0 else 0.0,
                mdd=float((s / s.cummax() - 1).min()))


say("样本: %d 天 x %d 币  %s .. %s" % (n, len(cols), dates[0].date(), dates[-1].date()))
say("费用: 单边 %.2f%%   调仓周期 = 每 N 天重算一次信号并全额换仓" % (FEE * 100))
say("")
hdr = "%-22s %6s %10s %9s %8s %9s" % ("模式 / 周期", "年换手", "总收益", "年化", "夏普", "最大回撤")
say(hdr)
say("-" * len(hdr))

fig, ax = plt.subplots(figsize=(13, 7))
for mname, fn in [("簇内传导(多空)", lambda i: sel_cluster(i)),
                  ("动量7d(多空)", lambda i: sel_mom(i, 7))]:
    for hold in (1, 7, 14, 30):
        s, to = run(fn, True, hold, FEE)
        a = stat(s)
        say("%-22s %6.0f %9.1f%% %8.1f%% %8.2f %8.1f%%"
            % ("%s %dd" % (mname, hold), to, a["tot"] * 100, a["cagr"] * 100, a["sh"], a["mdd"] * 100))
        ax.plot(s.index, s.values, lw=1.3, label="%s %dd" % (mname, hold))
    say("")

ax.set_yscale("log")
ax.set_title("调仓周期对净收益的影响（多空、扣费单边 0.10%）", fontsize=11)
ax.set_ylabel("NAV (log)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
plt.tight_layout()
p1 = os.path.join(EV, "backtest_freq.png")
plt.savefig(p1, dpi=125)
plt.close()
say("saved: %s" % p1)
with open(os.path.join(EV, "backtest_freq.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
