# -*- coding: utf-8 -*-
"""补充挖掘：① 资金费率套利的真实净收益  ② 更长历史的数据可得性

① 用 evidence/funding_scan.csv（Gate USDT 永续费率历史）
   算：扣调仓成本后的净年化、费率稳定性、风险提示
② 测币安现货最早可取时间（拉 BTCUSDT 2017 年数据看是否存在）
"""
import os, time, json
import numpy as np
import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EV = os.path.join(ROOT, 'evidence')
TXT = []
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); TXT.append(s)

t0 = time.time()
# ---------- ① 资金费率 ----------
say('=' * 90)
say('【①】资金费率套利：真实净收益（买现货 + 空永续）')
fp = os.path.join(EV, 'funding_scan.csv')
if os.path.exists(fp):
    d = pd.read_csv(fp)
    say('  funding_scan.csv 列：%s' % list(d.columns))
    say('  行数 = %d' % len(d))
    say(d.head(8).to_string())
    # 自动挑年化列
    ann_col = None
    for c in d.columns:
        if 'ann' in c.lower() or '年化' in c:
            ann_col = c; break
    if ann_col is None:
        for c in d.columns:
            if d[c].dtype.kind in 'fi':
                ann_col = c
    if ann_col:
        v = pd.to_numeric(d[ann_col], errors='coerce').dropna()
        if v.abs().median() < 1:      # 列是小数形式（0.176 = 17.6%）-> 转成百分数
            v = v * 100
        say('  用列 %s（已转百分数）：中位 %.2f%%  均值 %.2f%%  >10%% 个数 %d  >20%% 个数 %d' % (
            ann_col, v.median(), v.mean(), int((v > 10).sum()), int((v > 20).sum())))
        # 扣调仓成本：一年 12 次调仓（每月一次），单次往返 0.14%~0.30%
        for rt, lbl in ((0.14, '低成本(0.05%+0.02%单边)'), (0.19, '中(0.075%+0.02%)'), (0.30, '高(0.10%+0.05%)')):
            cost_year = rt * 12
            net = v - cost_year
            say('    %-22s 年成本 %.2f%% -> 净年化中位 %.2f%%  为正个数 %d/%d' % (
                lbl, cost_year, net.median(), int((net > 0).sum()), len(net)))
else:
    say('  funding_scan.csv 不存在')

# ---------- ② 更长历史 ----------
say('\n' + '=' * 90)
say('【②】更长历史：币安现货最早可取时间')
BASE = 'https://data-api.binance.vision/api/v3/klines'
for sym, start in (('BTCUSDT', '2017-08-17'), ('ETHUSDT', '2017-08-17'), ('BNBUSDT', '2017-11-06'),
                   ('SOLUSDT', '2020-08-11'), ('DOGEUSDT', '2019-07-05')):
    ms = int(pd.Timestamp(start, tz='UTC').timestamp() * 1000)
    try:
        r = requests.get(BASE, params={'symbol': sym, 'interval': '1d', 'startTime': ms, 'limit': 3},
                         timeout=20)
        js = r.json()
        if isinstance(js, list) and js:
            first = pd.to_datetime(js[0][0], unit='ms', utc=True)
            say('  %-10s 请求 %s -> 实际最早 %s  (%d 条)' % (
                sym, start, first.strftime('%Y-%m-%d'), len(js)))
        else:
            say('  %-10s 请求 %s -> 空/错误: %s' % (sym, start, str(js)[:80]))
    except Exception as e:
        say('  %-10s ERR %s' % (sym, str(e)[:70]))

say('\n  说明：能取到 2017 年 -> 「更长历史」这条路可行，但样本会偏向「老币」')
say('  而 502 币里多数是 2021 年后上市，拉 6 年只会让「有全史的币」更少')

open(os.path.join(EV, 'extra_mine.txt'), 'w', encoding='utf-8').write('\n'.join(TXT))
say('\nTOTAL %.0fs' % (time.time() - t0))
