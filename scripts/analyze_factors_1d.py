# -*- coding: utf-8 -*-
"""全量横截面结构：让因子自己浮现（不预设 BTC 主导）

输入：coinlab/data/klines_1d_top502.csv
做法：
  1) 取历史够长的币（>= MIN_BARS 根），对齐共同时间窗
  2) 相关矩阵 -> 特征分解 -> 平行分析定因子数（模拟数据对照）
  3) 检查 PC1 是不是「就是 BTC」
  4) 层次聚类切簇 -> 看簇是不是板块
输出：evidence/factors_1d.txt、evidence/factors_1d.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data")
EV = os.path.join(ROOT, "evidence")
os.makedirs(EV, exist_ok=True)

MIN_BARS = 800
N_NULL = 15

lines = []
def say(s=""):
    print(s, flush=True)
    lines.append(str(s))

df = pd.read_csv(os.path.join(DATA, "klines_1d_top502.csv"))
df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_localize(None)
cnt = df.groupby("symbol").size().sort_values()
say("file rows=%d  symbols=%d" % (len(df), df["symbol"].nunique()))
say("bars/symbol: min=%d  median=%d  max=%d" % (cnt.min(), int(cnt.median()), cnt.max()))
say("bars>= %d 的币数 = %d" % (MIN_BARS, int((cnt >= MIN_BARS).sum())))

keep = cnt[cnt >= MIN_BARS].index
sub = df[df["symbol"].isin(keep)]
px = sub.pivot_table(index="open_time", columns="symbol", values="close").sort_index()
px = px.dropna()
say("对齐后: 时间点=%d  币数=%d  区间 %s .. %s"
    % (len(px), px.shape[1], px.index[0].date(), px.index[-1].date()))

ret = px.pct_change().dropna()
say("样本=%d 币数=%d  (K 线数会远大于独立样本；见纪要 §7.2)" % ret.shape)

Z = (ret - ret.mean()) / ret.std().replace(0, np.nan)
Z = Z.dropna(axis=1, how="any")
cols = list(Z.columns)
X = Z.values
n, p = X.shape
C = np.corrcoef(X, rowvar=False)
vals = np.sort(np.linalg.eigvalsh(C))[::-1]

# 平行分析：同尺寸随机数据的特征值
rng = np.random.default_rng(20261015)
sim = np.array([np.sort(np.linalg.eigvalsh(np.corrcoef(rng.standard_normal((n, p)), rowvar=False)))[::-1]
                for _ in range(N_NULL)]).mean(axis=0)

nfac = int((vals > sim).sum())
say("")
say("=== 特征值谱 vs 平行分析（随机对照）===")
say("PC1=%.1f  PC2=%.1f  PC3=%.1f  PC4=%.1f  PC5=%.1f"
    % tuple(vals[:5]))
say("随机基线    PC1=%.1f  PC2=%.1f  PC3=%.1f  PC4=%.1f  PC5=%.1f"
    % tuple(sim[:5]))
say("PC1 解释方差 = %.1f%%  PC2 = %.1f%%  PC3 = %.1f%%"
    % (100 * vals[0] / p, 100 * vals[1] / p, 100 * vals[2] / p))
say("超过随机基线的因子数 = %d" % nfac)

# 特征向量
w, v = np.linalg.eigh(C)
order = np.argsort(w)[::-1]
w = w[order]; v = v[:, order]
pc = pd.DataFrame(v[:, :4], index=cols, columns=["PC1", "PC2", "PC3", "PC4"])

btc = ret["BTCUSDT"] if "BTCUSDT" in ret.columns else None
say("")
say("=== PC1 / PC2 是不是「就是 BTC」===")
for k in range(3):
    score = (Z[cols].values @ v[:, k])
    ser = pd.Series(score, index=Z.index)
    if btc is not None:
        d = pd.concat([ser.rename("pc"), btc.rename("btc")], axis=1).dropna()
        say("PC%d 与 BTC 日收益相关 = %+.3f" % (k + 1, d["pc"].corr(d["btc"])))
say("PC1 载荷最高的 10 个币: %s"
    % ", ".join("%s(%.2f)" % (s, pc.loc[s, "PC1"]) for s in pc["PC1"].abs().sort_values(ascending=False).head(10).index))
say("PC1 载荷最低的 10 个币: %s"
    % ", ".join("%s(%.2f)" % (s, pc.loc[s, "PC1"]) for s in pc["PC1"].abs().sort_values().head(10).index))

# 每币对 PC1 的解释力
pc1 = pd.Series(Z.values @ v[:, 0], index=Z.index)
r2 = {}
for c in cols:
    d = pd.concat([pc1.rename("p"), Z[c].rename("c")], axis=1).dropna()
    r2[c] = float(d["p"].corr(d["c"]) ** 2)
r2 = pd.Series(r2).sort_values()
say("")
say("各币对 PC1 的 R2: median=%.3f  <0.3 的币数=%d  <0.1 的币数=%d"
    % (r2.median(), int((r2 < 0.3).sum()), int((r2 < 0.1).sum())))
say("对 PC1 解释力最低的 15 个币: %s" % ", ".join(r2.head(15).index))

# 聚类
D = np.clip(1.0 - C, 0.0, None)
np.fill_diagonal(D, 0.0)
Zl = linkage(squareform(D, checks=False), method="average")
for thr in (0.7, 0.5, 0.35):
    lab = fcluster(Zl, t=thr, criterion="distance")
    sizes = pd.Series(lab).value_counts()
    big = sizes[sizes >= 5]
    say("")
    say("=== 距离阈值 %.2f -> %d 簇；>=5 个币的簇 %d 个 ===" % (thr, len(sizes), len(big)))
    for cid in big.index[:8]:
        members = [cols[i] for i in np.where(lab == cid)[0]]
        say("  簇%d (%d个): %s" % (cid, len(members), ", ".join(m.replace("USDT", "") for m in members[:22])
                                + (" ..." if len(members) > 22 else "")))

# --- 图 ---
fig = plt.figure(figsize=(15, 9))
ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(range(1, 21), vals[:20], "o-", color="0.2", label="actual")
ax1.plot(range(1, 21), sim[:20], "s--", color="0.7", label="random baseline")
ax1.set_title("Scree plot vs parallel analysis", fontsize=10)
ax1.set_xlabel("component"); ax1.set_ylabel("eigenvalue"); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

ax2 = fig.add_subplot(2, 2, 2)
ax2.scatter(pc["PC1"], pc["PC2"], s=10, color="0.4")
for name in ["BTCUSDT", "ETHUSDT", "TRXUSDT", "XAUTUSDT", "DOGEUSDT", "XRPUSDT"]:
    if name in pc.index:
        ax2.annotate(name.replace("USDT", ""), (pc.loc[name, "PC1"], pc.loc[name, "PC2"]), fontsize=8)
ax2.axhline(0, color="0.8", lw=0.8); ax2.axvline(0, color="0.8", lw=0.8)
ax2.set_title("PC1 vs PC2 loadings (%d coins)" % p, fontsize=10)
ax2.set_xlabel("PC1 loading"); ax2.set_ylabel("PC2 loading"); ax2.grid(alpha=0.3)

ax3 = fig.add_subplot(2, 1, 2)
show_n = min(70, p)
Cdf = pd.DataFrame(C, index=cols, columns=cols)
order_c = list(Cdf.sum(axis=1).sort_values(ascending=False).index[:show_n])
idx = [cols.index(c) for c in order_c]
Cs = C[np.ix_(idx, idx)]
im = ax3.imshow(Cs, cmap="gray_r", vmin=-0.2, vmax=1)
ax3.set_xticks(range(show_n)); ax3.set_xticklabels([cols[i].replace("USDT", "") for i in idx], rotation=90, fontsize=6)
ax3.set_yticks(range(show_n)); ax3.set_yticklabels([cols[i].replace("USDT", "") for i in idx], fontsize=6)
ax3.set_title("Correlation matrix, top %d coins by avg corr" % show_n, fontsize=10)
plt.colorbar(im, ax=ax3, fraction=0.02)
plt.tight_layout()
p1 = os.path.join(EV, "factors_1d.png")
plt.savefig(p1, dpi=125)
plt.close()

say("")
say("saved: %s" % p1)
with open(os.path.join(EV, "factors_1d.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
