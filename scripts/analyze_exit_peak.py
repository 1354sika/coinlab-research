"""
出场专项：'在最高峰左右完美退出' 到底值多少钱、要多准才值钱。

不训练任何模型，只测物理边界（回答用户的信念，而不是反驳它）：
  A. 峰值在哪：入场后第几根到顶（off_peak 分布）、峰值幅度分布、H 内截断率
  B. 精度成本：事后知道峰值、但必须晚 d 根才卖，收益衰减多少（d=0..12）
  C. 细粒度跟踪止盈 θ=0.1%~1.0%：均值/中位/季正 + 卖价离峰值多远（capture）
     —— capture 就是'离最高峰左右'的量化：0=卖在峰值，-1%=比峰值低 1%

入场同 V20：ret24<=-2% 且 volr>=3.0，5m，H=96。
"""
import numpy as np, pandas as pd, time

DATA = r'D:/中信证券/coinlab/data/klines_5m_3y.csv'
OUT = r'D:/中信证券/coinlab/evidence/exit_peak_REAL.txt'

H, GAP, COST, W = 96, 24, 0.0004, 48
DIP, VOLR = -0.02, 3.0
VALID_FROM_Q = 4

t0 = time.time()
df = pd.read_csv(DATA, usecols=['symbol', 'open_time', 'high', 'low', 'close', 'volume'])
print('rows %d load %.1fs' % (len(df), time.time() - t0))


def prep(g):
    g = g.sort_values('open_time')
    h = g.high.to_numpy(np.float32); l = g.low.to_numpy(np.float32)
    c = g.close.to_numpy(np.float32); v = g.volume.to_numpy(np.float32)
    t = g.open_time.to_numpy()
    n = len(c)
    if n < 600 + H:
        return None
    vma = pd.Series(v).rolling(24, min_periods=24).mean().to_numpy()
    volr = np.where(vma > 0, v / vma, 1.0)
    ret24 = np.zeros(n); ret24[24:] = c[24:] / c[:-24] - 1.0
    return dict(h=h, l=l, c=c, t=t, n=n, volr=volr, ret24=ret24)


paths = {}; entries = []
for sym, g in df.groupby('symbol', sort=False):
    P = prep(g)
    if P is None:
        continue
    paths[sym] = P
    for i in range(W + 300, P['n'] - H - 1, GAP):
        if P['ret24'][i] <= DIP and P['volr'][i] >= VOLR:
            entries.append((int(P['t'][i]), sym, i))
del df
entries.sort()
qm_all = pd.to_datetime([e[0] for e in entries], unit='ms').to_period('Q')
qs = sorted(set(qm_all.unique().tolist()))
print('好入场 %d  验证自 %s  %.1fs' % (len(entries), qs[VALID_FROM_Q], time.time() - t0))

rows = []
for ei, (tms, sym, i) in enumerate(entries):
    P = paths[sym]
    c, h, l = P['c'], P['h'], P['l']
    gp = c[i:i + H] / c[i] - 1.0
    eh = h[i:i + H] / c[i] - 1.0
    el = l[i:i + H] / c[i] - 1.0
    runmax = np.maximum.accumulate(eh)
    pk = int(np.argmax(eh))
    rows.append(dict(ei=ei, q=str(qm_all[ei]), gp=gp, eh=eh, el=el, runmax=runmax,
                     pk=pk, pk_ret=float(eh[pk]), n=len(gp)))
print('路径预计算完毕 %.1fs' % (time.time() - t0))

val = [r for r in rows if r['q'] >= str(qs[VALID_FROM_Q])]
print('验证集路径 %d' % len(val))
L = []
L.append('出场·峰值退出可达性实测（零模型，只测物理边界）')
L.append('入场: ret24<=%.0f%% 且 volr>=%.1f  |  H=96根(8h)  |  验证自 %s  |  成本 %.2f%%/笔(单边)'
         % (DIP * 100, VOLR, qs[VALID_FROM_Q], COST * 100))
L.append('样本: 验证集 %d 笔，%d 个季度' % (len(val), len(set(r['q'] for r in val))))
L.append('')

# A. 峰值位置与幅度
pk_off = np.array([r['pk'] for r in val])
pk_ret = np.array([r['pk_ret'] for r in val])
trunc = float((pk_off >= H - 3).mean())  # 峰落在窗口末尾 => 实际可能还没到顶
L.append('=== A. 峰值在哪（验证集 %d 笔）' % len(val))
L.append('  到顶时刻(根,5m)   中位 %d  P25 %d  P75 %d  P90 %d   （满窗 %d 根 = 8h）'
         % (np.median(pk_off), np.percentile(pk_off, 25), np.percentile(pk_off, 75),
            np.percentile(pk_off, 90), H))
L.append('  到顶时涨幅        均值 %.3f%%  中位 %.3f%%  P25 %.3f%%  P75 %.3f%%'
         % (pk_ret.mean() * 100, np.median(pk_ret) * 100,
            np.percentile(pk_ret, 25) * 100, np.percentile(pk_ret, 75) * 100))
L.append('  峰值未到(窗口末尾) 占比 %.1f%%  <- 真峰可能在 8h 之后，属右截断' % (trunc * 100))
L.append('  峰值幅度分布        <=0%%(入场即顶) %.1f%%   0~2%% %.1f%%   2~5%% %.1f%%   >5%% %.1f%%'
         % ((pk_ret <= 0).mean() * 100, ((pk_ret > 0) & (pk_ret <= 0.02)).mean() * 100,
            ((pk_ret > 0.02) & (pk_ret <= 0.05)).mean() * 100, (pk_ret > 0.05).mean() * 100))
L.append('')

# B. 精度成本：事后知峰，晚 d 根卖
L.append('=== B. 精度成本：事后知道峰值，但只能晚 d 根(×5分钟)才卖')
L.append('  %-8s %10s %10s %10s %10s' % ('延迟d(根)', '均值/笔', '中位/笔', '占峰值幅', '相对d=0'))
base = None
for d in (0, 1, 2, 3, 4, 6, 8, 12, 24):
    rets = []
    for r in val:
        k = min(r['pk'] + d, r['n'] - 1)
        rets.append(float(r['gp'][k]) - COST)
    rets = np.array(rets)
    cap = rets.mean() / (pk_ret.mean() - COST) * 100 if pk_ret.mean() else 0
    if base is None:
        base = rets.mean()
    L.append('  %-10d %9.3f%% %9.3f%% %9.1f%% %9.1f%%'
             % (d, rets.mean() * 100, np.median(rets) * 100, cap,
                rets.mean() / base * 100 if base else 0))
L.append('  注: d=0 是"用 high 的最高点成交"的理论上界（现实中要求卖单已在峰值价位挂着且成交）；')
L.append('      d=1 起才是"看到峰值后用下一根收盘卖出"，是可执行的上界。')
L.append('')

# C. 细粒度跟踪止盈
def trailing(theta):
    net, cap, offs = [], [], []
    for r in val:
        gp, eh, el, hi = r['gp'], r['eh'], r['el'], 0.0
        sell = r['n'] - 1
        for j in range(r['n']):
            if eh[j] > hi:
                hi = eh[j]
            dd = (1.0 + el[j]) / (1.0 + hi) - 1.0
            if j >= 1 and dd <= -theta:
                sell = j
                break
        net.append(float(gp[sell]) - COST)
        cap.append((1.0 + gp[sell]) / (1.0 + r['pk_ret']) - 1.0)
        offs.append(sell)
    return np.array(net), np.array(cap), np.array(offs)


L.append('=== C. 跟踪止盈细网格：越贴近峰值，是否越值钱')
L.append('  %-8s %10s %10s %8s %12s %10s' % ('θ', '均值/笔', '中位/笔', '季正', '离峰中位', '持有时长中位'))
for th in (0.001, 0.002, 0.003, 0.004, 0.005, 0.0075, 0.01, 0.015):
    net, cap, offs = trailing(th)
    qq = pd.Series([r['q'] for r in val])
    g = pd.DataFrame({'q': qq, 'net': net}).groupby('q').net.mean()
    L.append('  %-9s %9.3f%% %9.3f%% %6d/%d %11.2f%% %9d根'
             % ('%.2f%%' % (th * 100), net.mean() * 100, np.median(net) * 100,
                int((g > 0).sum()), len(g), np.median(cap) * 100, int(np.median(offs))))
L.append('')

# D. 结论所需的对照：完美程度上限 vs 现实可达
L.append('=== D. "完美退出"的价钱与落差')
oracle = pk_ret.mean() - COST
net1, cap1, _ = trailing(0.01)
L.append('  理论上界(卖在最高点, d=0)      %8.3f%%/笔' % (oracle * 100))
L.append('  可执行上界(晚1根收盘卖, d=1)   %8.3f%%/笔' % ((np.mean([float(r['gp'][min(r['pk'] + 1, r['n'] - 1)]) for r in val]) - COST) * 100))
L.append('  现实: 跟踪止盈 θ=1.0%%         %8.3f%%/笔  离峰中位 %.2f%%' % (net1.mean() * 100, np.median(cap1) * 100))
e3 = np.mean([float(r['gp'][min(11, r['n'] - 1)]) - COST for r in val])
L.append('  现实: 固定持 12 根            %8.3f%%/笔' % (e3 * 100))
L.append('  理论上界 / θ=1.0%% 现实  = %.1f 倍' % (oracle / net1.mean() if net1.mean() else 0))
L.append('')
L.append('诚实边界: 单边 maker %.2f%% 未核实往返；H=8h 右截断 %.1f%%；'
         '入场点 STEP=%d 导致样本重叠；零滑点、零成交率假设。'
         % (COST * 100, trunc * 100, GAP))

txt = '\n'.join(L)
print(txt)
open(OUT, 'w', encoding='utf-8').write(txt + '\n')
print('\n已写', OUT)
print('TOTAL %.0fs' % (time.time() - t0))
