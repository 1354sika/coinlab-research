# -*- coding: utf-8 -*-
"""9 年样本重跑（2017-08-17 ~ 2026-10-15）：判「簇内传导是行情还是规律」

与之前 2.2 年版本的关键差别：
  1. 样本从 811 天 -> 3350 天
  2. 动态币池：每天只用当天有数据的币（不 dropna 成共同窗），早期币少、后期币多
  3. 分年报告（这是判「行情 vs 规律」的关键）
诚实边界：币池 = 今天仍在 Binance 交易的前 502 名 -> 存在幸存者偏差（当年退市的币不在池内）
输出：evidence/backtest_long.txt / .png
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
LOOKBACK, REFIT, FEE, ANN = 120, 60, 0.001, 365
lines = []


def say(s=""):
    print(s, flush=True)
    lines.append(str(s))


df = pd.read_csv(os.path.join(DATA, "klines_1d_full.csv"))
df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_localize(None)
df = df.drop_duplicates(["symbol", "open_time"])
px = df.pivot_table(index="open_time", columns="symbol", values="close").sort_index()
ret = px.pct_change().iloc[1:]
dates, cols, n = list(ret.index), list(ret.columns), len(ret)
start = LOOKBACK + 1
say("样本: %d 天 x %d 币  %s .. %s" % (n, len(cols), dates[0].date(), dates[-1].date()))
say("活跃币数: 起 %d -> 末 %d" % (int(ret.iloc[0].notna().sum()), int(ret.iloc[-1].notna().sum())))

refits = []
for i in range(LOOKBACK, n, REFIT):
    win = ret.iloc[i - LOOKBACK:i]
    cnt = win.notna().sum()
    ok = [c for c in win.columns if cnt[c] >= LOOKBACK * 0.8]
    if len(ok) < 15:
        continue
    sub = win[ok].dropna()
    if len(sub) < LOOKBACK * 0.5:
        continue
    Z = ((sub - sub.mean()) / sub.std().replace(0, np.nan)).dropna(axis=1, how="any")
    if Z.shape[1] < 10:
        continue
    C = np.corrcoef(Z.values, rowvar=False)
    w, v = np.linalg.eigh(C)
    PC1 = pd.Series(v[:, np.argsort(w)[::-1][0]], index=list(Z.columns))
    D = np.clip(1.0 - C, 0.0, None)
    np.fill_diagonal(D, 0.0)
    lab = fcluster(linkage(squareform(D, checks=False), method="average"), t=0.5, criterion="distance")
    refits.append((i, PC1, pd.Series(lab, index=list(Z.columns))))
_starts = [r[0] for r in refits]
say("滚动重估: %d 次" % len(refits))


def get_ref(i):
    k = max(bisect.bisect_right(_starts, i - 1) - 1, 0)
    return refits[k][1], refits[k][2]


def sel_cluster(i, thr=0.05, k=3):
    _, lab = get_ref(i)
    r_prev = ret.iloc[i - 1].dropna()
    lo, sh = [], []
    for _, grp in lab.groupby(lab):
        if len(grp) < 3:
            continue
        syms = [s for s in grp.index if s in r_prev.index]
        rr = r_prev[syms]
        if len(rr) < 3 or rr.max() < thr:
            continue
        o = rr.sort_values()
        lo += list(o.index[:k])
        sh += list(o.index[-k:])
    return lo, sh


def sel_mom(i, lb=7, k=20):
    past = ret.iloc[max(0, i - lb):i]
    cum = ((1.0 + past).prod() - 1.0).dropna().sort_values(ascending=False)
    if len(cum) < 2 * k:
        return [], []
    return list(cum.index[:k]), list(cum.index[-k:])


def sel_factor(i, topn=30):
    pc1, _ = get_ref(i)
    a = pc1.abs().sort_values(ascending=False)
    return list(a.index[:topn]), list(a.index[-topn:])


def build(lo, sh, ls):
    w = pd.Series(dtype=float)
    lo = [s for s in lo if s in ret.columns]
    sh = [s for s in sh if s in ret.columns]
    if lo:
        w = pd.concat([w, pd.Series(0.5 / len(lo) if ls else 1.0 / len(lo), index=lo)])
    if ls and sh:
        w = pd.concat([w, pd.Series(-0.5 / len(sh), index=sh)])
    return w.groupby(level=0).sum()


def run(sig_fn, ls, hold, fee=FEE):
    nav, prev, c = 1.0, pd.Series(dtype=float), []
    w = pd.Series(dtype=float)
    for i in range(start, n):
        if (i - start) % hold == 0:
            lo, sh = sig_fn(i)
            w = build(lo, sh, ls)
        r = ret.iloc[i].fillna(0.0)
        idx = w.index.union(prev.index)
        t = float((w.reindex(idx).fillna(0.0) - prev.reindex(idx).fillna(0.0)).abs().sum())
        gross = float((w.reindex(r.index).fillna(0.0) * r).sum())
        nav *= (1.0 + gross - t * fee)
        c.append(nav)
        prev = w
    return pd.Series(c, index=dates[start:])


def stat(s):
    r = s.pct_change().dropna()
    yrs = len(r) / ANN
    return dict(tot=float(s.iloc[-1] - 1), cagr=float(s.iloc[-1] ** (1 / yrs) - 1),
                sh=float(r.mean() / r.std() * np.sqrt(ANN)) if r.std() > 0 else 0.0,
                mdd=float((s / s.cummax() - 1).min()))


MODES = [
    ("簇内传导(多空)", lambda i: sel_cluster(i), 1),
    ("动量7d(多空)", lambda i: sel_mom(i, 7), 7),
    ("动量30d(多空)", lambda i: sel_mom(i, 30), 7),
    ("因子暴露(多空)", lambda i: sel_factor(i), 7),
]

say("")
say("=== 9 年全期（扣费单边 %.2f%%）===" % (FEE * 100))
hdr = "%-18s %10s %9s %8s %9s" % ("模式", "总收益", "年化", "夏普", "最大回撤")
say(hdr)
say("-" * len(hdr))
curves = {}
for name, fn, hold in MODES:
    s = run(fn, True, hold)
    a = stat(s)
    curves[name] = s
    say("%-18s %9.1f%% %8.1f%% %8.2f %8.1f%%" % (name, a["tot"] * 100, a["cagr"] * 100, a["sh"], a["mdd"] * 100))

say("")
say("=== 分年（多空、扣费后年度收益%）===")
years = sorted({int(y) for s in curves.values() for y in s.index.year})
hdr2 = "%-18s" % "模式" + "".join("%8d" % y for y in years)
say(hdr2)
say("-" * len(hdr2))
yearly = {}
for name, fn, hold in MODES:
    s = curves[name]
    r = s.pct_change().dropna()
    row = []
    for y in years:
        ry = r[r.index.year == y]
        row.append(((1 + ry).prod() - 1) * 100 if len(ry) > 5 else float("nan"))
    yearly[name] = row
    say("%-18s" % name + "".join("%7.0f%%" % v if v == v else "%8s" % "-" for v in row))

say("")
say("判读：若某模式只有个别年份大赚、多数年份大亏 -> 行情依赖，不是规律")

# 基准：等权全市场（动态池）与 BTC
b_ew = run(lambda i: (list(ret.iloc[i - 1].dropna().index), []), False, 7)
a = stat(b_ew)
say("")
say("基准 等权全市场(动态池,满仓): 总收益 %+.1f%%  年化 %+.1f%%  夏普 %.2f" % (a["tot"] * 100, a["cagr"] * 100, a["sh"]))
b_btc = run(lambda i: (["BTCUSDT"], []), False, 30)
a = stat(b_btc)
say("基准 BTC 买入持有:          总收益 %+.1f%%  年化 %+.1f%%  夏普 %.2f" % (a["tot"] * 100, a["cagr"] * 100, a["sh"]))
curves["等权全市场"] = b_ew
curves["BTC"] = b_btc

fig, ax = plt.subplots(figsize=(13, 7))
for name, s in curves.items():
    ax.plot(s.index, s.values, lw=1.3, label=name)
ax.set_yscale("log")
ax.set_title("9 年样本（2017-08 ~ 2026-10）多空组合净值（扣费单边 0.10%）", fontsize=11)
ax.set_ylabel("NAV (log)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
plt.tight_layout()
p1 = os.path.join(EV, "backtest_long.png")
plt.savefig(p1, dpi=125)
plt.close()
say("")
say("saved: %s" % p1)
with open(os.path.join(EV, "backtest_long.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
