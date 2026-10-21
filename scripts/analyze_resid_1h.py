# -*- coding: utf-8 -*-
"""扣共同因子后的残差结构（纪要 §4.6 步2：先扣共同因子，否则全是 BTC 星形图）

输入：coinlab/data/klines_1h_top20_365d.csv
做法：每币 ret ~ a + b*BTC_ret + c*ETH_ret 全样本 OLS → 残差
     残差之间再算相关 → 若仍有高相关，说明存在 BTC/ETH 之外的共同结构（板块因子）
输出：evidence/resid_corr_1h.txt、evidence/resid_corr_1h.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
from scipy.spatial.distance import squareform

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data")
EV = os.path.join(ROOT, "evidence")
os.makedirs(EV, exist_ok=True)

df = pd.read_csv(os.path.join(DATA, "klines_1h_top20_365d.csv"))
df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_localize(None)
px = df.pivot_table(index="open_time", columns="symbol", values="close").sort_index()
ret = px.pct_change()

lines = []
def say(s=""):
    print(s, flush=True)
    lines.append(str(s))

factors = ["BTCUSDT", "ETHUSDT"]
say("bars=%d  %s .. %s" % (len(ret), ret.index[0], ret.index[-1]))

stats = []
resid = {}
for c in ret.columns:
    if c in factors:
        continue
    d = pd.concat([ret[c].rename("y"), ret[factors]], axis=1).dropna()
    if len(d) < 300:
        continue
    X = np.column_stack([np.ones(len(d)), d["BTCUSDT"].values, d["ETHUSDT"].values])
    y = d["y"].values
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ beta
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (r ** 2).sum() / ss_tot
    stats.append((c, beta[1], beta[2], r2, float(np.std(r)), float(np.std(y))))
    resid[c] = pd.Series(r, index=d.index)

st = pd.DataFrame(stats, columns=["symbol", "beta_BTC", "beta_ETH", "R2", "resid_vol", "total_vol"])
st = st.sort_values("R2", ascending=False)
say("")
say("=== 每币对 BTC/ETH 的回归（解释力 R2）===")
say(st.round(4).to_string(index=False))
say("")
say("R2: median=%.3f  min=%.3f  max=%.3f" % (st["R2"].median(), st["R2"].min(), st["R2"].max()))
say("R2 < 0.2 的币（BTC/ETH 解释不了）: %s"
    % ", ".join(st.loc[st["R2"] < 0.2, "symbol"].tolist()))

R = pd.DataFrame(resid).dropna()
C = R.corr()
n = len(C)
off = C.values[np.triu_indices(n, 1)]
say("")
say("=== 残差相关（扣掉 BTC+ETH 之后还剩多少共同运动）===")
say("pairs=%d   mean=%.4f  median=%.4f  p90=%.4f  max=%.4f"
    % (len(off), off.mean(), np.median(off), np.percentile(off, 90), off.max()))

rng = np.random.default_rng(20261015)
null_off = []
for _ in range(200):
    a, b = rng.choice(n, 2, replace=False)
    va, vb = C.columns[a], C.columns[b]
    sh = int(rng.integers(200, len(R) - 200))
    rolled = pd.Series(np.roll(R[vb].values, sh), index=R.index)
    null_off.append(float(pd.concat([R[va], rolled], axis=1).dropna().corr().iloc[0, 1]))
null_off = np.array(null_off)
say("noise baseline: mean=%.4f  p95(|.|)=%.4f" % (null_off.mean(), np.percentile(np.abs(null_off), 95)))

pairs = []
for i in range(n):
    for j in range(i + 1, n):
        pairs.append((C.columns[i], C.columns[j], C.iloc[i, j]))
pairs.sort(key=lambda x: -abs(x[2]))
say("")
say("=== 残差相关最强的 20 对 ===")
for a, b, v in pairs[:20]:
    say("  %-12s %-12s %+.4f" % (a, b, v))

D = np.clip(1.0 - C.values, 0.0, None)   # 距离矩阵必须非负
np.fill_diagonal(D, 0.0)
Z = linkage(squareform(D, checks=False), method="average")
order = leaves_list(Z)
Cs = C.iloc[order, order]

fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), gridspec_kw={"width_ratios": [3, 2]})
im = axes[0].imshow(Cs.values, cmap="gray_r", vmin=-1, vmax=1)
axes[0].set_xticks(range(n)); axes[0].set_xticklabels(Cs.columns, rotation=90, fontsize=7)
axes[0].set_yticks(range(n)); axes[0].set_yticklabels(Cs.columns, fontsize=7)
axes[0].set_title("Residual corr after removing BTC+ETH (1h, 365d)", fontsize=10)
plt.colorbar(im, ax=axes[0], fraction=0.046)
dendrogram(Z, labels=list(C.columns), orientation="right", ax=axes[1],
           color_threshold=0, above_threshold_color="0.3")
axes[1].set_title("Cluster tree (residual corr distance)", fontsize=10)
plt.tight_layout()
p1 = os.path.join(EV, "resid_corr_1h.png")
plt.savefig(p1, dpi=130)
plt.close()

say("")
say("saved: %s" % p1)
with open(os.path.join(EV, "resid_corr_1h.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
