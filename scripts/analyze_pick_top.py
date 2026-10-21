"""
用户口径已确认：投入 100 赚 150 = 单笔 +50%（满仓、无杠杆）。
他"不看所有币"——本脚本模拟"挑着看"：

每个决策时点，按某种排名挑出 TopN 个币，统计这些币在之后 8 小时内
max(high)/entry-1 >= 50% 的比例，与"全池随机"对比。

排名口径（都只用 K 线，符合他"只看 K 线"）：
  R1 24h 涨幅前 N
  R2 1h 涨幅前 N
  R3 量比前 N
  R4 24h 波动率（标准差）前 N
N = 3 / 5 / 10 / 20
"""
import numpy as np, pandas as pd, time

DATA = r'D:/中信证券/coinlab/data/klines_5m_3y.csv'
OUT = r'D:/中信证券/coinlab/evidence/pick_topN.txt'
H, STEP = 96, 96

t0 = time.time()
df = pd.read_csv(DATA, usecols=['symbol', 'open_time', 'high', 'low', 'close', 'volume'])

# 统一时间轴：按 open_time 建宽表
frames = []
for sym, g in df.groupby('symbol', sort=False):
    g = g.sort_values('open_time')
    if len(g) < 1200 + H:
        continue
    h = g.high.to_numpy(np.float32); l = g.low.to_numpy(np.float32)
    c = g.close.to_numpy(np.float32); v = g.volume.to_numpy(np.float32)
    t = g.open_time.to_numpy(); n = len(c)
    vma = pd.Series(v).rolling(24, min_periods=24).mean().to_numpy()
    volr = np.where((vma > 0) & np.isfinite(vma), v / vma, 1.0)
    ret24 = np.zeros(n); ret24[24:] = c[24:] / c[:-24] - 1.0
    ret12 = np.zeros(n); ret12[12:] = c[12:] / c[:-12] - 1.0
    lr = np.zeros(n); lr[1:] = np.log(c[1:] / c[:-1])
    sd96 = pd.Series(lr).rolling(96, min_periods=96).std().to_numpy()
    futmax = pd.Series(h).rolling(H, min_periods=H).max().shift(-H).to_numpy()
    frames.append(pd.DataFrame(dict(sym=sym, t=t, c=c, ret24=ret24, ret12=ret12,
                                    volr=volr, sd96=sd96, futmax=futmax, n=np.arange(n), N=n)))
del df
print('准备 %d 币 %.1fs' % (len(frames), time.time() - t0), flush=True)

# 只保留决策点（每 96 根），并丢弃无效
rows = []
for f in frames:
    m = (f['n'] % STEP == 0) & (f['n'] >= 600) & np.isfinite(f['futmax']) & (f['c'] > 0)
    sub = f[m]
    if len(sub) == 0:
        continue
    rows.append(pd.DataFrame(dict(
        t=sub['t'].values, sym=sub['sym'].values,
        mr=sub['futmax'].values / sub['c'].values - 1.0,
        ret24=sub['ret24'].values, ret12=sub['ret12'].values,
        volr=sub['volr'].values, sd96=sub['sd96'].values)))
D = pd.concat(rows, ignore_index=True)
print('决策点 %d  %.1fs' % (len(D), time.time() - t0), flush=True)

L = []
L.append('"挑着看"能否把 8h 内到 +50%% 的比例拉到一半？  %s' % time.strftime('%Y-%m-%d %H:%M'))
L.append('每个决策时点按排名取前 N 个币；指标 = 这些币 8h 内 max(high)/entry-1 >= 50%')
L.append('数据 60 币 / 3 年 / 5m；决策点每 8 小时一个')
L.append('')
L.append('  %-22s %8s %8s %10s %10s %10s' % ('排名口径', 'N', '样本数', '>=20%', '>=50%', '中位浮盈'))
L.append('  %-22s %8s %8d %9.1f%% %9.1f%% %9.1f%%'
         % ('全池（基准）', '-', len(D), (D.mr >= 0.2).mean() * 100,
            (D.mr >= 0.5).mean() * 100, D.mr.median() * 100))

for key, label, asc in (('ret24', 'R1 24h涨幅前N', True),
                        ('ret12', 'R2 1h涨幅前N', True),
                        ('volr', 'R3 量比前N', True),
                        ('sd96', 'R4 波动率前N', True)):
    for N in (3, 5, 10, 20):
        top = D.sort_values(['t', key], ascending=[True, False]).groupby('t').head(N)
        L.append('  %-22s %8d %8d %9.1f%% %9.1f%% %9.1f%%'
                 % (label, N, len(top), (top.mr >= 0.2).mean() * 100,
                    (top.mr >= 0.5).mean() * 100, top.mr.median() * 100))

L.append('')
L.append('诚实边界：')
L.append('  1. 排名用"当根已知"的 K 线量（24h涨幅/1h涨幅/量比/波动率），无前视。')
L.append('  2. 浮盈用 high 计算，成交假设见前；未计滑点/成本。')
L.append('  3. 60 币池是主流币；用户池若含小市值币，比例会更高（见 304 币池：单币最高 83%）。')

txt = '\n'.join(L)
print(txt)
open(OUT, 'w', encoding='utf-8').write(txt + '\n')
print('\n已写', OUT, 'TOTAL %.0fs' % (time.time() - t0))
