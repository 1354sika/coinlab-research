# -*- coding: utf-8 -*-
"""1m 尺度 lead-lag：分钟内是否存在 BTC 领先

输入：coinlab/data/klines_1m_top20_7d.csv
输出：evidence/leadlag_1m.txt、evidence/leadlag_1m.png

corr(BTC_ret(t-k), coin_ret(t))，k>0 表示 BTC 领先 k 分钟；lag 扫 -60..+60
噪声基线：随机移位该币收益序列 10 次，取 |peak| 中位数
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

df = pd.read_csv(os.path.join(DATA, "klines_1m_top20_7d.csv"))
df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_localize(None)
px = df.pivot_table(index="open_time", columns="symbol", values="close").sort_index()
ret = px.pct_change()

lines = []
def say(s=""):
    print(s, flush=True)
    lines.append(str(s))

say("bars=%d  symbols=%d  %s .. %s" % (len(px), px.shape[1], px.index[0], px.index[-1]))

btc = ret["BTCUSDT"]
LAGS = list(range(-60, 61))
others = [c for c in ret.columns if c != "BTCUSDT"]


def xcurve(coin, lead, max_lag=60):
    out = {}
    for k in range(-max_lag, max_lag + 1):
        d = pd.concat([lead.shift(k).rename("a"), coin.rename("b")], axis=1).dropna()
        out[k] = float(d["a"].corr(d["b"])) if len(d) > 300 else np.nan
    return pd.Series(out)


rows = []
curves = {}
rng = np.random.default_rng(20261015)
for c in others:
    s = xcurve(ret[c], btc)
    curves[c] = s
    v = s.dropna()
    k_star = int(v.abs().idxmax())
    same = float(s.get(0, np.nan))
    # 滞后 1~5 分钟中的最大值（看"短时领先"是否比同期强）
    short = {k: s[k] for k in (1, 2, 3, 4, 5) if k in s}
    best_short_k = max(short, key=lambda k: abs(short[k])) if short else None

    arr = ret[c].values.astype(float)
    null_peaks = []
    for _ in range(10):
        sh = int(rng.integers(500, max(501, len(arr) - 500)))
        s2 = xcurve(pd.Series(np.roll(arr, sh), index=ret[c].index), btc)
        v2 = s2.dropna()
        if len(v2):
            null_peaks.append(abs(float(v2.loc[v2.abs().idxmax()])))
    rows.append((c, same, k_star, float(s[k_star]), best_short_k,
                 float(short[best_short_k]) if short else np.nan,
                 float(np.median(null_peaks)) if null_peaks else np.nan))

res = pd.DataFrame(rows, columns=["symbol", "corr_same_m", "peak_lag_m", "peak_corr",
                                  "best_short_k", "best_short_corr", "null_peak_med"])
res["short_minus_same"] = res["best_short_corr"] - res["corr_same_m"]
res = res.sort_values("corr_same_m", ascending=False)

say("")
say("=== 1m 尺度：与 BTC 的互相关（lag -60..+60 分钟）===")
say(res.round(4).to_string(index=False))
say("")
say("peak_lag_m 分布: %s" % dict(res["peak_lag_m"].value_counts().sort_index()))
say("同期 1m 相关: mean=%.4f median=%.4f max=%.4f"
    % (res["corr_same_m"].mean(), res["corr_same_m"].median(), res["corr_same_m"].max()))
say("噪声 |peak| 中位数 的均值 = %.4f" % res["null_peak_med"].mean())
say("真实 |peak| 超过噪声基线的个数 = %d / %d"
    % (int((res["peak_corr"].abs() > res["null_peak_med"]).sum()), len(res)))
say("『滞后1-5分钟』相关 高于 同期相关 的币数 = %d / %d"
    % (int((res["short_minus_same"] > 0).sum()), len(res)))

fig, ax = plt.subplots(figsize=(9, 5))
for c in [x for x in ["ETHUSDT", "SOLUSDT", "DOGEUSDT", "XRPUSDT", "PEPEUSDT", "AVAXUSDT",
                      "ENAUSDT", "WALUSDT"] if x in curves]:
    ax.plot(curves[c].index, curves[c].values, lw=1.1, label=c)
ax.axvline(0, color="0.6", lw=0.8, ls="--")
ax.set_xlabel("lag in minutes   (>0 : BTC leads)")
ax.set_ylabel("corr(BTC_ret(t-lag), coin_ret(t))")
ax.set_title("1m lead-lag vs BTC (7d, top20 USDT pairs)", fontsize=11)
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
plt.tight_layout()
p1 = os.path.join(EV, "leadlag_1m.png")
plt.savefig(p1, dpi=130)
plt.close()

fig, ax = plt.subplots(figsize=(8, 5))
order = res.sort_values("peak_lag_m")
ax.barh(order["symbol"], order["peak_lag_m"], color="0.55")
ax.axvline(0, color="0.2", lw=0.9)
ax.set_xlabel("peak lag (minutes)   >0 : BTC leads")
ax.set_title("Peak lead-lag vs BTC, 1m bars (7d)", fontsize=11)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
p2 = os.path.join(EV, "peaklag_1m.png")
plt.savefig(p2, dpi=130)
plt.close()

say("")
say("saved: %s" % p1)
say("saved: %s" % p2)
with open(os.path.join(EV, "leadlag_1m.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
