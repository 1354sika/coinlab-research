# -*- coding: utf-8 -*-
"""用户两问：① 有没有「长收益」的信号；② 币能不能预测自己（多阶结构）

之前只测过一阶自相关 AC(1)（滤波后 +0.16），没测多阶结构，也没测长周期信号。
本脚本：
  A) 多阶自相关 AC(k)，k=1..30（分钟级，两种价格口径），看结构是否单调衰减
  B) AR(k) 样本外预测力：用过去 k 分钟预测下一分钟，walk-forward，看 R² 与扣费后收益
  C) 长周期：日线 AC(k)，k=1..20（502 币 9 年），看有没有跨日的自相关
数据：klines_1m_sync.csv（分钟）+ klines_1d_full.csv（日线）
输出：evidence/selfpred.txt / .png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data")
EV = os.path.join(ROOT, "evidence")
lines = []


def say(s=""):
    print(s, flush=True)
    lines.append(s)


# ---------- A/B 分钟级 ----------
df = pd.read_csv(os.path.join(DATA, "klines_1m_sync.csv"),
                 usecols=["symbol", "open_time", "high", "low", "close"])
C = df.pivot(index="open_time", columns="symbol", values="close").sort_index()
MID = ((df.pivot(index="open_time", columns="symbol", values="high"))
       + (df.pivot(index="open_time", columns="symbol", values="low"))) / 2.0
Vc = C.pct_change().iloc[1:].to_numpy(dtype=np.float32)
Vm = MID.pct_change().iloc[1:].to_numpy(dtype=np.float32)
syms = list(C.columns)
T, N = Vc.shape
say("分钟矩阵 %d x %d" % (T, N))

say("")
say("=== A) 多阶自相关 AC(k)（分钟级，100 币的中位）===")
say("%-6s %14s %14s" % ("k", "收盘价 AC", "(高+低)/2 AC"))
lags = [1, 2, 3, 5, 10, 15, 30, 60]
ac_c, ac_m = {}, {}
for k in lags:
    a1, a2 = [], []
    for j in range(N):
        c = Vc[:, j]
        m = np.isfinite(c[:-k]) & np.isfinite(c[k:])
        if m.sum() > 2000:
            a1.append(np.corrcoef(c[:-k][m], c[k:][m])[0, 1])
        d = Vm[:, j]
        m2 = np.isfinite(d[:-k]) & np.isfinite(d[k:])
        if m2.sum() > 2000:
            a2.append(np.corrcoef(d[:-k][m2], d[k:][m2])[0, 1])
    ac_c[k] = np.nanmedian(a1)
    ac_m[k] = np.nanmedian(a2)
    say("%-6d %14.4f %14.4f" % (k, ac_c[k], ac_m[k]))

# ---------- B) AR(k) walk-forward ----------
say("")
say("=== B) AR(k) 样本外预测力（用过去 k 分钟预测下一分钟，walk-forward）===")
say("%-8s %10s %12s %12s %12s" % ("k", "样本数", "OOS R²", "方向命中率", "扣费后净收益/次%"))
t0 = 30000
for k in [1, 5, 30]:
    err_num, err_den, hit, n = 0.0, 0.0, 0, 0
    pnl = 0.0
    fee = 0.0002
    for j in range(60):          # 前 60 币
        X = Vm[:, j]
        tr = slice(20000, t0)
        te = slice(t0, T - 1)
        # 用过去估计 AR 系数（滚动简化：一次估计）
        y_tr = X[20000 + k:t0]
        Xm = np.column_stack([X[20000 + k - i - 1:t0 - i - 1] for i in range(k)])
        ok = np.isfinite(y_tr) & np.all(np.isfinite(Xm), axis=1)
        if ok.sum() < 2000:
            continue
        beta, *_ = np.linalg.lstsq(Xm[ok], y_tr[ok], rcond=None)
        y_te = X[t0 + k:T]
        Xt = np.column_stack([X[t0 + k - i - 1:T - i - 1] for i in range(k)])
        ok2 = np.isfinite(y_te) & np.all(np.isfinite(Xt), axis=1)
        if ok2.sum() < 500:
            continue
        pred = Xt[ok2] @ beta
        real = y_te[ok2]
        base = np.nanmean(real)
        err_num += np.sum((real - pred) ** 2)
        err_den += np.sum((real - base) ** 2)
        pos = np.sign(pred)
        hit += int(np.sum(pos == np.sign(real)))
        n += len(real)
        pnl += float(np.sum(pos * real)) - fee * len(real)
    if n:
        say("%-8d %10d %11.4f%% %11.4f%% %11.4f%%"
            % (k, n, (1 - err_num / max(err_den, 1e-12)) * 100, 100 * hit / n, pnl * 100 / n))

# ---------- C) 日线多阶 ----------
say("")
say("=== C) 日线多阶自相关 AC(k)（502 币 9 年，取共同窗）===")
d1 = pd.read_csv(os.path.join(DATA, "klines_1d_full.csv"), usecols=["symbol", "open_time", "close"])
d1["open_time"] = pd.to_datetime(d1["open_time"], unit="ms", utc=True).dt.tz_localize(None)
d1 = d1.drop_duplicates(["symbol", "open_time"])
P1 = d1.pivot(index="open_time", columns="symbol", values="close").sort_index()
R1 = P1.pct_change().iloc[1:]
say("%-6s %12s %12s" % ("k(天)", "AC中位", ">0的币数"))
dlags = [1, 2, 3, 5, 10, 20]
acd = {}
for k in dlags:
    vals = []
    for c in R1.columns:
        s = R1[c].dropna()
        if len(s) < 300:
            continue
        a = s[:-k].to_numpy()
        b = s[k:].to_numpy()
        vals.append(np.corrcoef(a, b)[0, 1])
    v = np.array(vals)
    acd[k] = np.nanmedian(v)
    say("%-6d %12.4f %12d" % (k, np.nanmedian(v), int((v > 0).sum())))

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
ax = axes[0]
ax.plot(lags, [ac_c[k] for k in lags], "o-", label="收盘价")
ax.plot(lags, [ac_m[k] for k in lags], "s--", label="(高+低)/2")
ax.axhline(0, color="0.8", lw=1)
ax.set_xscale("log")
ax.set_xlabel("k（分钟）")
ax.set_ylabel("AC(k) 中位")
ax.set_title("分钟级多阶自相关", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax = axes[1]
ax.plot(dlags, [acd[k] for k in dlags], "o-", color="0.25")
ax.axhline(0, color="0.8", lw=1)
ax.set_xlabel("k（天）")
ax.set_ylabel("AC(k) 中位")
ax.set_title("日线多阶自相关（9年）", fontsize=10)
ax.grid(alpha=0.3)
ax = axes[2]
ax.scatter(lags, [ac_m[k] for k in lags], s=30, color="0.4", label="分钟")
ax.axhline(0, color="0.8", lw=1)
ax.set_xscale("log")
ax.set_xlabel("k")
ax.set_ylabel("AC")
ax.set_title("(高+低)/2 口径自相关衰减", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
plt.tight_layout()
p1 = os.path.join(EV, "selfpred.png")
plt.savefig(p1, dpi=125)
plt.close()
say("")
say("saved: %s" % p1)
with open(os.path.join(EV, "selfpred.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
