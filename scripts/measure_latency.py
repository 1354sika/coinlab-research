# -*- coding: utf-8 -*-
"""measure_latency.py — 实测"看到信号"到底延迟几秒

测三件事：
  1) HTTP RTT：一次请求从发出到收到要多久
  2) 数据滞后：服务器返回的最新 1s K 线，比本地当前时间晚多少
  3) 端到端：RTT + 滞后 + live_recorder 的轮询间隔（2 秒）
"""
import time, json, urllib.request, statistics

BASE = 'https://data-api.binance.vision'
SYMS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT', 'XRPUSDT']
N = 12


def get(url, timeout=10):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode('utf-8'))


L = []
def p(s=''):
    print(s, flush=True)
    L.append(s)


p('=' * 88)
p('网络延迟实测  %s' % time.strftime('%Y-%m-%d %H:%M:%S'))
p('=' * 88)
p('')
p('  %-10s %10s %10s %12s %10s' % ('symbol', 'RTT(ms)', '数据滞后(s)', '最新K线时间', '服务器时间'))
p('  ' + '-' * 68)

rtts, lags = [], []
for s in SYMS:
    for i in range(N):
        t0 = time.time()
        try:
            d = get(BASE + '/api/v3/klines?symbol=%s&interval=1s&limit=3' % s)
        except Exception as e:
            p('  %-10s FAIL %s' % (s, str(e)[:40]))
            continue
        t1 = time.time()
        rtt = (t1 - t0) * 1000.0
        if not d:
            continue
        last = d[-1]
        close_ms = int(last[6])
        srv_ms = int(d[-1][0]) + 60000 if False else None
        now_ms = time.time() * 1000.0
        lag = (now_ms - close_ms) / 1000.0
        rtts.append(rtt)
        lags.append(lag)
        if i == 0:
            p('  %-10s %10.1f %10.2f %12s %10s'
              % (s, rtt, lag,
                 time.strftime('%H:%M:%S', time.gmtime(close_ms / 1000.0)),
                 time.strftime('%H:%M:%S', time.gmtime(now_ms / 1000.0))))
        time.sleep(0.3)

if not rtts:
    p('  全部失败，无法测量'); raise SystemExit

p('')
p('=== 汇总（%d 次请求）' % len(rtts))
p('  HTTP RTT      中位 %6.1f ms   均值 %6.1f ms   P90 %6.1f ms   最小 %6.1f   最大 %6.1f'
  % (statistics.median(rtts), statistics.mean(rtts),
     sorted(rtts)[int(len(rtts) * 0.9)], min(rtts), max(rtts)))
p('  数据滞后      中位 %6.2f s    均值 %6.2f s    P90 %6.2f s    最小 %6.2f   最大 %6.2f'
  % (statistics.median(lags), statistics.mean(lags),
     sorted(lags)[int(len(lags) * 0.9)], min(lags), max(lags)))

p('')
p('=== 端到端延迟拆解（看到"某秒跌了 1%%"需要多久）')
p('  ① live_recorder 轮询间隔            2.00 s（代码里 time.sleep(2)）')
p('  ② HTTP RTT（拉数据）                 %.2f s' % (statistics.median(rtts) / 1000.0))
p('  ③ 数据滞后（K 线收盘到可见）          %.2f s' % statistics.median(lags))
p('  -------------------------------------------------------')
p('  合计（最坏）= 一次完整的"从价格动到你看见"  %.2f s'
  % (2.0 + statistics.median(rtts) / 1000.0 + statistics.median(lags)))
p('')
p('  注意：如果信号是"1 秒跌 1%%"，你看到的其实是已经过去 %.1f 秒的价格。'
  % (2.0 + statistics.median(rtts) / 1000.0 + statistics.median(lags)))
p('  在这 %.1f 秒里价格又走了多少，才是真正的"延迟成本"。'
  % (2.0 + statistics.median(rtts) / 1000.0 + statistics.median(lags)))

with open(r'D:\中信证券\coinlab\evidence\latency.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(L))
print('written latency.txt')
