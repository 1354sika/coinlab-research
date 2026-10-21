# -*- coding: utf-8 -*-
"""v6：两个之前没测的算子 —— 条件传导 / 非线性尾部依赖

v4/v5 测的都是「无条件、线性、均值」口径，全部为负。本版补两个真正的盲区：
  1) 条件传导：只在特定市场状态下测「谁带动谁」
     - 市场大跌日（全市场收益 <= 5 分位）
     - 高波动期 / 低波动期（30 日滚动波动相对中位）
     - 全样本（对照）
  2) 非线性：Spearman 秩相关 vs Pearson 线性相关（差最大 = 关系最非线性）
  3) 尾部依赖：下尾共现比 P(A跌&B跌) / (P(A跌)P(B跌))，>1 表示「一起崩」
数据：9 年日线，121 个长期币
输出：evidence/cond_nonlin.txt / .png
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
MIN_BARS, Q = 2000, 0.97
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
mkt = ret.mean(axis=1)
ex = ret.sub(mkt, axis=0)
V = ex[big].to_numpy(dtype=float)
valid = np.isfinite(V)
T, N = V.shape
say("长期币 %d 个；样本 %d 天（%s .. %s）" % (N, len(ret), ret.index[0].date(), ret.index[-1].date()))

# --- 状态掩码 ---
vol30 = mkt.rolling(30).std()
states = {
    "全样本": np.ones(T, dtype=bool),
    "市场大跌日(<=5%)": (mkt <= mkt.quantile(0.05)).fillna(False).to_numpy(),
    "高波动期": (vol30 > vol30.median()).fillna(False).to_numpy(),
    "低波动期": (vol30 <= vol30.median()).fillna(False).to_numpy(),
}
for k, m in states.items():
    say("  状态[%s] 天数 = %d" % (k, int(m.sum())))


def scan_state(mask, h=1, min_common=80, min_ev=12):
    rows = []
    for a in range(N):
        for b in range(N):
            if a == b:
                continue
            nxt = np.zeros(T, dtype=bool)
            nxt[:-h] = valid[h:, b]
            common = valid[:, a] & valid[:, b] & mask & nxt
            idx = np.where(common)[0]
            if len(idx) < min_common:
                continue
            sa = V[idx, a]
            rb = V[idx + h, b]
            thr = np.quantile(sa, Q)
            m = sa > thr
            if m.sum() < min_ev:
                continue
            rows.append(dict(src=big[a], dst=big[b], lift=rb[m].mean() - rb.mean(),
                             n_ev=int(m.sum()), n_common=len(idx)))
    return pd.DataFrame(rows)


say("")
say("=== 1) 条件传导：各市场状态下「源冲击 -> 次日响应」的 lift 分布 ===")
say("%-18s %8s %11s %11s %11s %10s" % ("状态", "对数", "中位%", "均值%", "|lift|>1%对数", "最强lift%"))
res = {}
for sname, mask in states.items():
    d = scan_state(mask)
    res[sname] = d
    if len(d) == 0:
        say("%-18s %8d  (样本不足)" % (sname, 0))
        continue
    say("%-18s %8d %10.3f %10.3f %11d %10.3f"
        % (sname, len(d), d["lift"].median() * 100, d["lift"].mean() * 100,
           int((d["lift"].abs() > 0.01).sum()), d["lift"].abs().max() * 100))

for sname, d in res.items():
    if len(d) == 0:
        continue
    say("")
    say("  [%s] |lift| 最大的 8 对：" % sname)
    for r_ in d.reindex(d["lift"].abs().sort_values(ascending=False).index).head(8).itertuples():
        say("    %-13s -> %-13s %+7.3f%%  (冲击%d次/共同%d天)"
            % (r_.src, r_.dst, r_.lift * 100, r_.n_ev, r_.n_common))

# --- 2) 非线性：Spearman vs Pearson ---
say("")
say("=== 2) 非线性：Spearman 秩相关 vs Pearson 线性相关（超额收益）===")
E = ex[big]
pe = E.corr()
sp = E.rank().corr()
iu = np.triu_indices(N, k=1)
pv = pe.to_numpy()[iu]
sv = sp.to_numpy()[iu]
say("Pearson 中位 %.3f；Spearman 中位 %.3f" % (np.nanmedian(pv), np.nanmedian(sv)))
say("Spearman - Pearson 的中位差 %.4f；|差|>0.05 的币对数 %d / %d"
    % (np.nanmedian(sv - pv), int((np.abs(sv - pv) > 0.05).sum()), len(pv)))
diff = pd.DataFrame({"p": pv, "s": sv, "d": sv - pv})
say("  Spearman 明显高于 Pearson（非线性单调关系更强）的 top 8：")
for i in diff["d"].abs().sort_values(ascending=False).head(8).index:
    a, b = iu[0][i], iu[1][i]
    say("    %-13s %-13s Pearson %+.3f  Spearman %+.3f  差 %+.3f"
        % (big[a], big[b], pv[i], sv[i], sv[i] - pv[i]))

# --- 3) 尾部依赖 ---
say("")
say("=== 3) 尾部依赖：下尾共现比（>1 = 一起崩）===")
low = (E < E.quantile(0.10)).astype(float)
n = low.shape[0]
co = low.T.values @ low.values / n
marg = low.mean().to_numpy()
expm = np.outer(marg, marg)
ratio = co / np.where(expm > 0, expm, np.nan)
np.fill_diagonal(ratio, np.nan)
rv = ratio[iu]
say("下尾共现比：中位 %.2f，>1.5 的币对数 %d / %d" % (np.nanmedian(rv), int(np.nansum(rv > 1.5)), len(rv)))
say("  （若完全独立，该比值应=1.00）")
order = np.argsort(-np.nan_to_num(rv))[:8]
say("  共现比最高的 8 对：")
for i in order:
    a, b = iu[0][i], iu[1][i]
    say("    %-13s %-13s 共现比 %.2f" % (big[a], big[b], rv[i]))

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
ax = axes[0]
for sname, d in res.items():
    if len(d):
        ax.hist(d["lift"] * 100, bins=70, alpha=0.45, label=sname)
ax.axvline(0, color="0.2", lw=1)
ax.set_title("各市场状态下的 lift 分布（%）", fontsize=10)
ax.legend(fontsize=7)
ax.grid(alpha=0.3)
ax = axes[1]
ax.scatter(pv, sv, s=2, color="0.45")
ax.plot([-0.2, 1], [-0.2, 1], color="0.75", lw=1)
ax.set_xlabel("Pearson"); ax.set_ylabel("Spearman")
ax.set_title("线性 vs 秩相关（偏离对角线=非线性）", fontsize=10)
ax.grid(alpha=0.3)
ax = axes[2]
ax.hist(np.clip(rv, 0, 3), bins=80, color="0.55")
ax.axvline(1.0, color="0.2", lw=1, ls="--")
ax.set_title("下尾共现比分布（虚线=独立）", fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
p1 = os.path.join(EV, "cond_nonlin.png")
plt.savefig(p1, dpi=125)
plt.close()
say("")
say("saved: %s" % p1)
with open(os.path.join(EV, "cond_nonlin.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
