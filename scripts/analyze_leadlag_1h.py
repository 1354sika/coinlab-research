# -*- coding: utf-8 -*-
"""1h 尺度 lead-lag：BTC 相对各币是领先还是滞后，领先几小时

输入：coinlab/data/klines_1h_top20_365d.csv
输出：evidence/leadlag_1h.txt、evidence/leadlag_1h.png

方法（对照纪要 §4.6 步1、§8.6 对照组的简化版）：
  corr(BTC_ret(t-k), coin_ret(t))，k>0 表示 BTC 领先 k 小时
  同时用「随机移位 20 次」构造噪声基线，看真实峰值是否超出噪声水平
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

csv_path = os.path.join(DATA, "klines_1h_top20_365d.csv")
df = pd.read_csv(csv_path)
df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_localize(None)
px = df.pivot_table(index="open_time", columns="symbol", values="close").sort_index()
ret = px.pct_change()

lines = []
def say(s=""):
    print(s, flush=True)
    lines.append(str(s))

say("bars=%d  symbols=%d  %s .. %s" % (len(px), px.shape[1], px.index[0], px.index[-1]))
say("BTC 缺失小时数 = %d" % int(ret["BTCUSDT"].isna().sum()))

btc = ret["BTCUSDT"]
LAGS = list(range(-24, 25))
others = [c for c in ret.columns if c != "BTCUSDT"]


def xcurve(coin, lead, max_lag=24):
    out = {}
    for k in range(-max_lag, max_lag + 1):
        d = pd.concat([lead.shift(k).rename("a"), coin.rename("b")], axis=1).dropna()
        out[k] = float(d["a"].corr(d["b"])) if len(d) > 200 else np.nan
    return pd.Series(out)


def peak_of_series(v, lags):
    s = pd.Series(v, index=lags).dropna()
    if s.empty:
        return np.nan, np.nan
    k = s.abs().idxmax()
    return int(k), float(s[k])


rows = []
curves = {}
rng = np.random.default_rng(20261015)
for c in others:
    s = xcurve(ret[c], btc)
    curves[c] = s
    k_star, r_star = peak_of_series(s.values, LAGS)
    same = float(s.get(0, np.nan))
    # 噪声基线：随机移位该币的收益序列，重复 20 次取 |peak| 的中位数
    v = ret[c].values.astype(float)
    null_peaks = []
    for _ in range(20):
        sh = int(rng.integers(500, max(501, len(v) - 500)))
        rolled = pd.Series(np.roll(v, sh), index=ret[c].index)
        s2 = xcurve(rolled, btc)
        _, r2 = peak_of_series(s2.values, LAGS)
        if not np.isnan(r2):
            null_peaks.append(abs(r2))
    null_med = float(np.median(null_peaks)) if null_peaks else np.nan
    rows.append((c, same, k_star, r_star, null_med))

res = pd.DataFrame(rows, columns=["symbol", "corr_same_h", "peak_lag_h", "peak_corr", "null_peak_med"])
res["peak_minus_null"] = res["peak_corr"].abs() - res["null_peak_med"]
res = res.sort_values("corr_same_h", ascending=False)

say("")
say("=== 与 BTC 的 1h 互相关 ===")
say("corr_same_h = 同期1小时相关; peak_lag_h>0 表示 BTC 领先该币那么多小时")
say("null_peak_med = 随机移位 20 次的 |峰值| 中位数（噪声基线）")
say(res.round(4).to_string(index=False))
say("")
say("peak_lag_h 分布: %s" % dict(res["peak_lag_h"].value_counts().sort_index()))
say("同期 1h 相关: mean=%.3f median=%.3f max=%.3f"
    % (res["corr_same_h"].mean(), res["corr_same_h"].median(), res["corr_same_h"].max()))
say("真实|peak| 超过噪声基线的个数 = %d / %d"
    % (int((res["peak_minus_null"] > 0).sum()), len(res)))

# --- 图 1：几个代表币的互相关曲线 ---
show = [c for c in ["ETHUSDT", "SOLUSDT", "DOGEUSDT", "XRPUSDT", "PEPEUSDT", "TRXUSDT"] if c in curves]
fig, ax = plt.subplots(figsize=(9, 5))
for c in show:
    ax.plot(curves[c].index, curves[c].values, lw=1.2, label=c)
ax.axvline(0, color="0.6", lw=0.8, ls="--")
ax.set_xlabel("lag in hours  (>0 : BTC leads)")
ax.set_ylabel("corr(BTC_ret(t-lag), coin_ret(t))")
ax.set_title("1h lead-lag vs BTC (365d, top20 USDT pairs)", fontsize=11)
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
plt.tight_layout()
p1 = os.path.join(EV, "leadlag_1h.png")
plt.savefig(p1, dpi=130)
plt.close()

# --- 图 2：峰值滞后 分布 ---
fig, ax = plt.subplots(figsize=(8, 5))
order = res.sort_values("peak_lag_h")
ax.barh(order["symbol"], order["peak_lag_h"], color="0.55")
ax.axvline(0, color="0.2", lw=0.9)
ax.set_xlabel("peak lag (hours)   >0 : BTC leads")
ax.set_title("Peak lead-lag vs BTC, 1h bars", fontsize=11)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
p2 = os.path.join(EV, "peaklag_1h.png")
plt.savefig(p2, dpi=130)
plt.close()

say("")
say("saved: %s" % p1)
say("saved: %s" % p2)
with open(os.path.join(EV, "leadlag_1h.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
