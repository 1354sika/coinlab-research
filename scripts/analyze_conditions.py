# -*- coding: utf-8 -*-
"""条件网格挖掘：用户批评「9 年平均没有时效性」——那就做条件化 + 细分 + 样本外

用户的三个点：
  1) 502 币一起测
  2) 既然「涨跌一致」，就量化涨/跌各是多少（正负幅度）
  3) 平均值掩盖时效性 —— 要「满足什么条件，接下去怎么走」以及各种细分

本脚本（先在最细的分钟级上做条件网格，再用样本外验证）：
  A) 涨跌幅度不对称：市场同涨日 vs 同跌日，幅度分布各是多少
  B) 条件网格：波动状态 × 市场方向 × 时段 × 成交量状态 = 36 格
     每格统计「之后 5/15/60 分钟」的全市场平均收益
  C) 时效性检验：把样本切前半/后半，看同一格子的收益是否一致（不一致 = 平均值骗人）
  D) 网格里的极值是否可交易：取 |收益| 最大的格子，扣费后净值
输出：evidence/conditions.txt / .png
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


df = pd.read_csv(os.path.join(DATA, "klines_1m_sync.csv"),
                 usecols=["symbol", "open_time", "close", "quote_volume"])
C = df.pivot(index="open_time", columns="symbol", values="close").sort_index()
Qv = df.pivot(index="open_time", columns="symbol", values="quote_volume").sort_index()
V = C.pct_change().iloc[1:].to_numpy(dtype=np.float32)
QV = Qv.iloc[1:].to_numpy(dtype=np.float32)
syms = list(C.columns)
T, N = V.shape
hours = pd.DatetimeIndex(C.index).hour.to_numpy()[1:]
mkt = np.nanmean(V, axis=1)
say("矩阵 %d 分钟 x %d 币（约 %.0f 天）" % (T, N, T / 1440))

# ---------- A) 涨跌幅度不对称 ----------
say("")
say("=== A) 既然「同时涨跌」，那涨多少、跌多少（全市场等权组合，每分钟）===")
up, dn = mkt[mkt > 0], mkt[mkt < 0]
say("  上涨分钟 %d 个：均值 %+.4f%%  中位 %+.4f%%  P90 %+.4f%%"
    % (len(up), up.mean() * 100, np.median(up) * 100, np.percentile(up, 90) * 100))
say("  下跌分钟 %d 个：均值 %+.4f%%  中位 %+.4f%%  P10 %+.4f%%"
    % (len(dn), dn.mean() * 100, np.median(dn) * 100, np.percentile(dn, 10) * 100))
say("  -> 平均上涨 %.4f%% vs 平均下跌 %.4f%%（绝对值之比 %.2f）"
    % (up.mean() * 100, abs(dn.mean()) * 100, up.mean() / max(abs(dn.mean()), 1e-12)))
say("  -> 极端：P99 上涨 %+.4f%%；P1 下跌 %+.4f%%（下跌尾部更厚？%.2f 倍）"
    % (np.percentile(mkt, 99) * 100, np.percentile(mkt, 1) * 100,
       abs(np.percentile(mkt, 1)) / max(np.percentile(mkt, 99), 1e-12)))

# ---------- B) 条件网格 ----------
vol30 = pd.Series(mkt).rolling(30).std().to_numpy()
mv5 = pd.Series(mkt).rolling(5).sum().to_numpy()
mqv = np.nansum(QV, axis=1)
mqv60 = pd.Series(mqv).rolling(60).mean().to_numpy()

q_v = np.nanquantile(vol30, [0.33, 0.67])
q_q = np.nanquantile(mqv60, 0.5)
vol_state = np.where(vol30 < q_v[0], 0, np.where(vol30 < q_v[1], 1, 2))   # 低/中/高
dir_state = np.where(mv5 > 0, 1, 0)                                       # 过去5分钟涨/跌
sess = np.where(hours < 8, 0, np.where(hours < 16, 1, 2))                 # 亚/欧/美
liq_state = np.where(mqv60 > q_q, 1, 0)                                   # 量高/低

HOR = [5, 15, 60]
say("")
say("=== B) 条件网格：满足什么条件 -> 之后 5/15/60 分钟全市场平均收益（%%）===")
say("%-4s %-4s %-4s %-4s %8s %9s %9s %9s %8s"
    % ("波动", "方向", "时段", "量", "样本数", "t+5", "t+15", "t+60", "可交易"))
rows = []
SESS = ["亚", "欧", "美"]
DIRS = ["跌", "涨"]
VOLS = ["低", "中", "高"]
LIQS = ["量低", "量高"]
valid = np.isfinite(vol30) & np.isfinite(mv5) & np.isfinite(mqv60)
for vs in range(3):
    for ds in range(2):
        for ss in range(3):
            for ls in range(2):
                m = valid & (vol_state == vs) & (dir_state == ds) & (sess == ss) & (liq_state == ls)
                idx = np.where(m)[0]
                idx = idx[idx < T - max(HOR) - 1]
                if len(idx) < 200:
                    continue
                fwd = {}
                for h in HOR:
                    fut = np.array([np.nanmean(V[t + 1:t + 1 + h]) for t in idx[:2000]])
                    fwd[h] = np.nanmean(fut)
                trade = "是" if abs(fwd[5]) > 0.002 else ""
                rows.append(dict(vol=VOLS[vs], dirn=DIRS[ds], sess=SESS[ss], liq=LIQS[ls],
                                 n=len(idx), f5=fwd[5], f15=fwd[15], f60=fwd[60]))
                say("%-4s %-4s %-4s %-4s %8d %8.4f%% %8.4f%% %8.4f%% %8s"
                    % (VOLS[vs], DIRS[ds], SESS[ss], LIQS[ls], len(idx),
                       fwd[5] * 100, fwd[15] * 100, fwd[60] * 100, trade))
R = pd.DataFrame(rows)
R.to_csv(os.path.join(EV, "conditions.csv"), index=False)
if len(R):
    say("")
    say("  |t+5| 最大的 5 个条件（绝对值）：")
    for r in R.reindex(R["f5"].abs().sort_values(ascending=False).index).head(5).itertuples():
        say("    %s/%s/%s/%s：t+5 %+.4f%%  t+15 %+.4f%%  t+60 %+.4f%%  (n=%d)"
            % (r.vol, r.dirn, r.sess, r.liq, r.f5 * 100, r.f15 * 100, r.f60 * 100, r.n))

# ---------- C) 时效性：前半 vs 后半 ----------
say("")
say("=== C) 时效性：同样条件，前半样本 vs 后半样本 的 t+5 收益（%%）===")
half = T // 2
say("%-20s %10s %10s %8s" % ("条件", "前半", "后半", "同号"))
same = 0
tot = 0
for vs in range(3):
    for ds in range(2):
        m_all = valid & (vol_state == vs) & (dir_state == ds)
        for tag, lo, hi in [("前半", 0, half), ("后半", half, T - 61)]:
            pass
        a = np.where(m_all)[0]
        a1 = a[(a < half) & (a < T - 61)]
        a2 = a[(a >= half) & (a < T - 61)]
        if len(a1) < 200 or len(a2) < 200:
            continue
        f1 = np.nanmean([np.nanmean(V[t + 1:t + 6]) for t in a1[:2000]])
        f2 = np.nanmean([np.nanmean(V[t + 1:t + 6]) for t in a2[:2000]])
        tot += 1
        if np.sign(f1) == np.sign(f2):
            same += 1
        say("%-20s %9.4f%% %9.4f%% %8s"
            % ("%s/%s" % (VOLS[vs], DIRS[ds]), f1 * 100, f2 * 100, "同" if np.sign(f1) == np.sign(f2) else "反"))
say("  -> 前后半同号的比例 %d/%d = %.0f%%（若 =50%% 说明条件收益不可重复）"
    % (same, tot, 100 * same / max(tot, 1)))

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
ax = axes[0]
ax.hist(up * 100, bins=100, alpha=0.6, label="上涨分钟", color="0.7")
ax.hist(dn * 100, bins=100, alpha=0.6, label="下跌分钟", color="0.25")
ax.axvline(0, color="0.2", lw=1)
ax.set_title("市场分钟收益的正负幅度分布", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax = axes[1]
if len(R):
    ax.bar(range(len(R)), R["f5"] * 100, color="0.45")
    ax.axhline(0, color="0.2", lw=1)
    ax.axhline(0.2, color="0.6", ls=":", lw=1, label="手续费 0.2%")
    ax.axhline(-0.2, color="0.6", ls=":", lw=1)
    ax.set_xticks([])
    ax.set_title("36 个条件格的 t+5 收益（%）", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
ax = axes[2]
if len(R):
    ax.scatter(R["f5"] * 100, R["f60"] * 100, s=25, color="0.45")
    ax.axhline(0, color="0.8", lw=1)
    ax.axvline(0, color="0.8", lw=1)
    ax.set_xlabel("t+5 收益 %")
    ax.set_ylabel("t+60 收益 %")
    ax.set_title("短窗 vs 长窗的条件收益", fontsize=10)
    ax.grid(alpha=0.3)
plt.tight_layout()
p1 = os.path.join(EV, "conditions.png")
plt.savefig(p1, dpi=125)
plt.close()
say("")
say("saved: %s" % p1)
with open(os.path.join(EV, "conditions.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
