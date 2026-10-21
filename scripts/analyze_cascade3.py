# -*- coding: utf-8 -*-
"""全量传导 v5：v4 没测掉的三件事（多日滞后 / 时变领先 / 路径传导）

v4 已否定「日线、t+1、直接对」口径下的币带动币（p<0.001 实际 0 对 vs 期望 14.5）。
本版把三个更「深」的通道补上：
  A) 多日滞后：t+1, t+2, t+3, t+5（传导可能需要几天才走完）
  B) 时变领先：把 9 年切 3 段，各段独立估 lift，看 A->B 是否在某段强、另一段反号
     （真正的「带动」应在某个时期稳定存在，而不是全样本被平均掉）
  C) 二阶路径：A 冲击 -> C 次日响应 -> B 后日响应（间接传导）
输出：evidence/cascade3.txt / .png
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
MIN_BARS, MIN_COMMON, Q, MIN_EV, NPERM = 2000, 800, 0.97, 25, 200
HORS = [1, 2, 3, 5]
lines = []


def say(s=""):
    print(s, flush=True)
    lines.append(str(s))


df = pd.read_csv(os.path.join(DATA, "klines_1d_full.csv"))
df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_localize(None)
df = df.drop_duplicates(["symbol", "open_time"])
px = df.pivot_table(index="open_time", columns="symbol", values="close").sort_index()
ret = px.pct_change()
ret = ret.loc[ret.notna().sum(axis=1) >= 5]
cnt = ret.notna().sum()
big = list(cnt[cnt >= MIN_BARS].index)
say("长期币 %d 个；样本 %d 天" % (len(big), len(ret)))
ex = ret.sub(ret.mean(axis=1), axis=0)
V = ex[big].to_numpy(dtype=float)
valid = np.isfinite(V)
T, N = V.shape
rng = np.random.default_rng(20261015)


def scan(h, idx_lo=0, idx_hi=None, topk=None):
    idx_hi = T - h if idx_hi is None else idx_hi
    rows = []
    for a in range(N):
        for b in range(N):
            if a == b:
                continue
            nxt = np.zeros(T, dtype=bool)
            nxt[:-h] = valid[h:, b]
            common = valid[:, a] & valid[:, b] & nxt
            common[:idx_lo] = False
            common[idx_hi:] = False
            idx = np.where(common)[0]
            if len(idx) < MIN_COMMON:
                continue
            sa = V[idx, a]
            rb = V[idx + h, b]
            m = sa > np.quantile(sa, Q)
            k = int(m.sum())
            if k < MIN_EV:
                continue
            obs = rb[m].mean() - rb.mean()
            rows.append(dict(src=big[a], dst=big[b], lift=obs, n_ev=k, n_common=len(idx)))
    return pd.DataFrame(rows)


say("")
say("=== A) 多日滞后扫描（每对独立检验，只报分布量级）===")
say("%-8s %8s %12s %12s %10s" % ("滞后", "对数", "lift中位%", "lift均值%", "|lift|>0.5%对数"))
res = {}
for h in HORS:
    d = scan(h)
    res[h] = d
    say("%-8s %8d %11.3f %11.3f %10d"
        % ("t+%d" % h, len(d), d["lift"].median() * 100, d["lift"].mean() * 100,
           int((d["lift"].abs() > 0.005).sum())))

# B) 时变：把样本切 3 段，看同一对的方向是否一致
say("")
say("=== B) 时变领先：9 年切 3 段，各段 lift 的符号一致性 ===")
seg = T // 3
seg_res = []
for s_i in range(3):
    d = scan(1, idx_lo=s_i * seg, idx_hi=(s_i + 1) * seg)
    d = d.rename(columns={"lift": "lift%d" % (s_i + 1)})
    seg_res.append(d.set_index(["src", "dst"]))
M = seg_res[0].join([seg_res[1][["lift2"]], seg_res[2][["lift3"]]], how="inner")
say("三段都有足够样本的币对: %d" % len(M))
if len(M):
    for c in ["lift1", "lift2", "lift3"]:
        say("  %s: 均值 %+.3f%%  中位 %+.3f%%" % (c, M[c].mean() * 100, M[c].median() * 100))
    same = ((M["lift1"] > 0) & (M["lift2"] > 0) & (M["lift3"] > 0)) | \
           ((M["lift1"] < 0) & (M["lift2"] < 0) & (M["lift3"] < 0))
    say("  三段同号的币对: %d / %d (%.0f%%)  [随机期望 25%%]"
        % (int(same.sum()), len(M), 100 * same.mean()))
    strong = M[(M[["lift1", "lift2", "lift3"]].abs().min(axis=1) > 0.005)]
    say("  三段每段 |lift| 都 > 0.5%% 的币对: %d" % len(strong))
    if len(strong):
        say("  %-13s %-13s %9s %9s %9s" % ("源", "响应", "段1%", "段2%", "段3%"))
        for (s_, d_), r_ in strong.sort_values("lift1", key=abs, ascending=False).head(15).iterrows():
            say("  %-13s %-13s %8.3f %8.3f %8.3f"
                % (s_, d_, r_["lift1"] * 100, r_["lift2"] * 100, r_["lift3"] * 100))
    M.reset_index().to_csv(os.path.join(EV, "cascade3_segments.csv"), index=False)

# C) 二阶路径
say("")
say("=== C) 二阶路径：A 冲击 -> C(t+1) -> B(t+2)，路径强度 ===")
d1 = res[1].set_index(["src", "dst"])["lift"]
d2 = res[2].set_index(["src", "dst"])["lift"]
both = pd.concat([d1.rename("a2c"), d2.rename("c2b")], axis=1, join="inner").dropna()
say("可用于路径计算的边: %d（一阶 %d 条）" % (len(both), len(d1)))
chain = both[(both["a2c"].abs() > 0.005) & (both["c2b"].abs() > 0.005)]
say("两侧都 > 0.5%% 的边: %d" % len(chain))

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
ax = axes[0]
for h in HORS:
    ax.hist(res[h]["lift"] * 100, bins=80, alpha=0.45, label="t+%d" % h)
ax.axvline(0, color="0.2", lw=1)
ax.set_title("多日滞后的 lift 分布（%）", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax = axes[1]
if len(M):
    ax.scatter(M["lift1"] * 100, M["lift2"] * 100, s=4, color="0.45")
    ax.axhline(0, color="0.8", lw=1)
    ax.axvline(0, color="0.8", lw=1)
    lim = np.nanpercentile(M[["lift1", "lift2"]].values * 100, 99)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
ax.set_title("段1 vs 段2 的 lift（若稳定应沿对角线）", fontsize=10)
ax.set_xlabel("段1 lift %")
ax.set_ylabel("段2 lift %")
ax.grid(alpha=0.3)
ax = axes[2]
ax.hist(both["a2c"] * 100, bins=80, color="0.55")
ax.axvline(0, color="0.2", lw=1)
ax.set_title("一阶边 a->c 的 lift 分布", fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
p1 = os.path.join(EV, "cascade3.png")
plt.savefig(p1, dpi=125)
plt.close()
say("")
say("saved: %s" % p1)
with open(os.path.join(EV, "cascade3.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
