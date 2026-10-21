# -*- coding: utf-8 -*-
"""v0 体检 + 最小结构观察

输入：coinlab/data/klines_1d_top50.csv
输出：evidence/health_v0.txt、evidence/leadlag_v0.png、evidence/btc_price_v0.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data")
EV = os.path.join(ROOT, "evidence")
os.makedirs(EV, exist_ok=True)

csv_path = os.path.join(DATA, "klines_1d_top50.csv")
df = pd.read_csv(csv_path)
df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_localize(None)
df = df.sort_values(["symbol", "open_time"])

lines = []
def say(s=""):
    print(s, flush=True)
    lines.append(str(s))

say("rows=%d  symbols=%d" % (len(df), df["symbol"].nunique()))
say("date range: %s .. %s" % (df["open_time"].min().date(), df["open_time"].max().date()))

g = df.groupby("symbol")
health = pd.DataFrame({
    "bars": g.size(),
    "first": g["open_time"].min().dt.date,
    "last": g["open_time"].max().dt.date,
    "zero_vol": g["volume"].apply(lambda s: int((s <= 0).sum())),
    "quote_vol_sum_bn": (g["quote_volume"].sum() / 1e9).round(2),
}).sort_values("bars")

say("")
say("=== 每符号体检（按 bars 升序）===")
say(health.to_string())

px = df.pivot_table(index="open_time", columns="symbol", values="close").sort_index()
ret = px.pct_change()
btc = ret["BTCUSDT"] if "BTCUSDT" in ret.columns else ret.iloc[:, 0]
btc_name = "BTCUSDT" if "BTCUSDT" in ret.columns else ret.columns[0]
say("")
say("基准 = %s, 交易日 = %d" % (btc_name, btc.notna().sum()))


def xcorr(lead, follow, k):
    """corr(lead.shift(k), follow)：k>0 表示 lead 领先 follow k 天"""
    d = pd.concat([lead.shift(k).rename("a"), follow.rename("b")], axis=1).dropna()
    if len(d) < 60:
        return np.nan, len(d)
    return float(d["a"].corr(d["b"])), len(d)


rows = []
for c in ret.columns:
    if c == btc_name:
        continue
    same, n0 = xcorr(btc, ret[c], 0)
    l1, n1 = xcorr(btc, ret[c], 1)
    l2, n2 = xcorr(btc, ret[c], 2)
    rows.append((c, n0, same, l1, l2))
res = pd.DataFrame(rows, columns=["symbol", "n", "corr_same", "btc_lead1", "btc_lead2"])
res["lead1_minus_same"] = res["btc_lead1"] - res["corr_same"]
res = res.sort_values("corr_same", ascending=False)

say("")
say("=== 与 %s 的相关（同期 / BTC领先1日 / 领先2日），按同期相关降序 ===" % btc_name)
say(res.round(4).to_string(index=False))

say("")
say("=== 同期相关分布 ===")
say("n=%d  mean=%.3f  median=%.3f  min=%.3f  max=%.3f  >0.7的个数=%d"
    % (res["corr_same"].notna().sum(), res["corr_same"].mean(),
       res["corr_same"].median(), res["corr_same"].min(), res["corr_same"].max(),
       int((res["corr_same"] > 0.7).sum())))

with open(os.path.join(EV, "health_v0.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

# --- 图 1：同期相关 vs BTC领先1日 相关（top20 by corr_same）---
top = res.head(20).iloc[::-1]
fig, ax = plt.subplots(figsize=(9, 7))
y = np.arange(len(top))
ax.barh(y + 0.2, top["corr_same"], height=0.4, label="same-day corr", color="0.35")
ax.barh(y - 0.2, top["btc_lead1"], height=0.4, label="BTC leads 1d", color="0.75")
ax.set_yticks(y)
ax.set_yticklabels(top["symbol"], fontsize=8)
ax.set_xlabel("correlation of daily returns")
ax.set_title("Same-day vs BTC-leads-1-day correlation (top50 USDT pairs)", fontsize=11)
ax.legend(fontsize=9)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
p1 = os.path.join(EV, "leadlag_v0.png")
plt.savefig(p1, dpi=130)
plt.close()

# --- 图 2：BTC 价格 + 日收益 ---
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                         gridspec_kw={"height_ratios": [2, 1]})
axes[0].plot(px.index, px[btc_name], lw=1.1, color="0.2")
axes[0].set_title("%s daily close (%s)" % (btc_name, "data-api.binance.vision"), fontsize=11)
axes[0].grid(alpha=0.3)
axes[1].plot(ret.index, ret[btc_name], lw=0.6, color="0.4")
axes[1].set_title("daily return", fontsize=9)
axes[1].grid(alpha=0.3)
plt.tight_layout()
p2 = os.path.join(EV, "btc_price_v0.png")
plt.savefig(p2, dpi=130)
plt.close()

say("")
say("saved: %s" % p1)
say("saved: %s" % p2)
with open(os.path.join(EV, "health_v0.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
