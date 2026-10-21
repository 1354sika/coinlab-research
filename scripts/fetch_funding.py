# -*- coding: utf-8 -*-
"""结构性套利可行性实测之一：资金费率到底有多少钱

数据源：Gate.io 公开 API（本机实测可达；注意其为本工作区可访问的观测口径，
        不代表可在该所合规交易——合规边界见调研报告）
取数：
  1) USDT 永续合约列表 + 24h 成交额
  2) 成交额 top N 合约的历史资金费率（limit 1000 期，约 333 天）
算：
  - 平均资金费率 / 年化
  - 为正的比例（做「买现货+空永续」时资金费是净收入的比例）
  - 与「往返手续费」比较：现货 taker + 合约 taker（列两档假设）
输出：evidence/funding_scan.txt / .csv
"""
import json
import os
import time

import numpy as np
import pandas as pd
import requests

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EV = os.path.join(ROOT, "evidence")
os.makedirs(EV, exist_ok=True)
G = "https://api.gateio.ws/api/v4"
TOPN = 60
lines = []


def say(s=""):
    print(s, flush=True)
    lines.append(str(s))


s = requests.Session()
r = s.get(G + "/futures/usdt/contracts", timeout=25)
cons = r.json()
say("Gate USDT 永续合约数 = %d" % len(cons))

rows = []
for c in cons:
    rows.append(dict(contract=c.get("name"), vol24=float(c.get("trade_size") or 0) * float(c.get("last_price") or 0),
                     funding_rate=float(c.get("funding_rate") or 0),
                     interval=int(c.get("funding_interval") or 28800),
                     price=float(c.get("last_price") or 0)))
meta = pd.DataFrame(rows).sort_values("vol24", ascending=False)
meta.to_csv(os.path.join(EV, "funding_meta.csv"), index=False)
say("合约类型 funding_interval 分布: %s" % meta["interval"].value_counts().to_dict())

out = []
for k, row in enumerate(meta.head(TOPN).itertuples(), 1):
    name = row.contract
    try:
        h = s.get(G + "/futures/usdt/funding_rate",
                  params=dict(contract=name, limit=1000), timeout=25).json()
    except Exception as e:
        say("  FAIL %s %s" % (name, str(e)[:60]))
        continue
    if not isinstance(h, list) or len(h) < 10:
        continue
    rr = np.array([float(x["r"]) for x in h])
    per_day = 86400 / max(row.interval, 1)
    ann = rr.mean() * per_day * 365
    out.append(dict(contract=name, n=len(rr), mean_rate=rr.mean(),
                    ann_mean=ann, pos_share=float((rr > 0).mean()),
                    ann_pos=rr[rr > 0].mean() * per_day * 365 if (rr > 0).any() else 0.0,
                    last=rr[-1], interval=row.interval))
    if k % 10 == 0:
        say("  %d/%d 最近 %s n=%d ann=%.1f%%" % (k, TOPN, name, len(rr), ann * 100))
    time.sleep(0.15)

d = pd.DataFrame(out).sort_values("ann_mean", ascending=False)
d.to_csv(os.path.join(EV, "funding_scan.csv"), index=False)

say("")
say("=== 资金费率年化（买现货+空永续的毛收益，%）top 20 ===")
say("%-14s %6s %10s %10s %9s %9s" % ("合约", "样本数", "年化均值", "年化(只算正)", "正比例", "最近一期"))
for row in d.head(20).itertuples():
    say("%-14s %6d %9.1f%% %9.1f%% %8.0f%% %8.3f%%"
        % (row.contract, row.n, row.ann_mean * 100, row.ann_pos * 100,
           row.pos_share * 100, row.last * 100))

say("")
say("=== 成本对照（往返，单边假设）===")
for spot, fut in [(0.001, 0.0005), (0.00075, 0.0002), (0.0005, 0.0002)]:
    rt = (spot + fut) * 2
    say("现货 %.3f%% + 合约 %.3f%% 单边 -> 往返 %.3f%%；若按 30 天持有一次调仓，年化成本 %.1f%%"
        % (spot * 100, fut * 100, rt * 100, rt * 12 * 100))

say("")
pos = int((d["ann_mean"] > 0).sum())
say("样本内 %d 个合约中，资金费年化均值为正的 %d 个（%.0f%%）" % (len(d), pos, 100 * pos / max(len(d), 1)))
say("年化均值 > 10%% 的合约数 = %d" % int((d["ann_mean"] > 0.10).sum()))
with open(os.path.join(EV, "funding_scan.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
say("")
say("saved: evidence/funding_scan.txt / .csv")
