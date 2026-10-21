# -*- coding: utf-8 -*-
"""gambler_h3_clean · h3 修掉前视后的真实水平

修法（与旧版 gambler_momentum_baseline 的三处不同）：
  1. 每笔先算自己的 t_close（触TP/触SL/超时 的确切时刻）
  2. 按 t_enter 排序后遍历；只有 heap 中 "t_close <= 本笔 t_enter" 的笔才计入 h3
     （旧版按 find_lows 的顺序遍历，而该顺序的时间不单调 → 会吸入未来的笔）
  3. h3 改用「最近3次里至少1次止盈」= h3_any

口径（对应用户 2026-10-17 决定）：
  入场 = 低点 × 1.03
  止盈 = 低点 × 1.30   （相对入场约 +26.2%）
  止损 = 入场 × 0.88   （相对入场 -12%，用户新定）
  持有上限 = 1440 分钟（24h）
  成本 = 0.50%/笔

数据：klines_1m_1y.csv（D01，60币/1年/1m）；按日历 75/25 切训练/测试
"""
import os, time, heapq
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

DATA = r'D:\中信证券\coinlab\data'
OUT = r'D:\中信证券\coinlab\evidence\cycle'
FEE = 0.005
START = 50.0
POS = 0.10
MAXSYM = int(os.environ.get('SYMS', '60'))

VW = 240          # V05 低点窗口
CONF = 2880       # V09 确认窗口
HOLD = 1440       # V16 持有上限
TPM = 0.30        # V14 止盈：低点×1.30
SLM = 0.12        # 止损：入场×0.88（-12%，用户新定）

BASE_FEATS = ['volr', 'tbr', 'v240', 'pos', 'lo_age', 'r1', 'r15', 'r60', 'hr', 'dw']
FEATS = BASE_FEATS + ['h3', 'h10', 'h3_any', 'n_hist']

logbuf = []
def log(s):
    print(s, flush=True); logbuf.append(s)

t0 = time.time()
log('=' * 100)
log('gambler_h3_clean · h3 修掉前视后的真实水平')
log('=' * 100)
log(f'入场 低点×{1.03:.2f}  止盈 低点×{1+TPM:.2f}  止损 入场×{1-SLM:.2f}  持有 {HOLD}min  费 {FEE:.2%}')
log(f'V05={VW}min  V09={CONF}min  修法：t_close<=t_enter 才计入 h3')

use = ['symbol', 'open_time', 'open', 'high', 'low', 'close', 'volume', 'taker_base']
RAW = {}
for ch in pd.read_csv(os.path.join(DATA, 'klines_1m_1y.csv'), usecols=use, chunksize=2_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        if sym not in RAW: RAW[sym] = []
        RAW[sym].append(g)
    if len(RAW) >= MAXSYM: break
RAWS = {}
for sym, parts in RAW.items():
    g = pd.concat(parts, ignore_index=True).sort_values('open_time').reset_index(drop=True)
    if len(g) < 5000: continue
    RAWS[sym] = (g['open'].values.astype(float), g['high'].values.astype(float),
                 g['low'].values.astype(float), g['close'].values.astype(float),
                 g['volume'].values.astype(float), g['taker_base'].values.astype(float),
                 g['open_time'].values.astype('int64'))
RAW = None
log(f'币 {len(RAWS)}  读取 {time.time()-t0:.0f}s')


def find_lows(l, c, n):
    lo_w = pd.Series(l).rolling(121, min_periods=1).min().values
    lo_i = []
    i = 120
    while i < n - CONF - 1:
        if l[i] <= lo_w[i]:
            lo_i.append(i); i += 30
        else:
            i += 1
    pairs = []
    for i in lo_i:
        if i + CONF >= n: continue
        for j in range(i + 1, min(i + CONF + 1, n)):
            if c[j] >= l[i] * 1.03:
                pairs.append((i, j)); break
    return pairs


def build(pairs, o, h, l, c, t, v, tb):
    """两阶段：先算每笔结果，再按 t_enter 排序做无前视的 h3"""
    n = len(c)
    rows = []
    for li, j in pairs:
        if j - VW < 0 or j + HOLD >= n: continue
        e = c[j]; lo_p = l[li]
        tp_px = lo_p * (1.0 + TPM)
        sl_px = e * (1.0 - SLM)          # 相对入场 -12%
        seg_h = h[j + 1:j + 1 + HOLD]
        seg_l = l[j + 1:j + 1 + HOLD]
        hit_tp = seg_h >= tp_px
        hit_sl = seg_l <= sl_px
        i_tp = int(np.argmax(hit_tp)) if hit_tp.any() else -1
        i_sl = int(np.argmax(hit_sl)) if hit_sl.any() else -1
        if i_tp >= 0 and i_sl >= 0:
            if i_tp < i_sl:
                ret = tp_px / e - 1; is_tp = 1; i_exit = i_tp
            else:
                ret = sl_px / e - 1; is_tp = 0; i_exit = i_sl
        elif i_tp >= 0:
            ret = tp_px / e - 1; is_tp = 1; i_exit = i_tp
        elif i_sl >= 0:
            ret = sl_px / e - 1; is_tp = 0; i_exit = i_sl
        else:
            ret = c[min(j + HOLD, n - 1)] / e - 1; is_tp = 0; i_exit = HOLD - 1

        t_enter = int(t[j])
        t_close = int(t[j + 1 + i_exit])

        vw = v[j - VW:j + 1]
        v240 = float(np.std(np.log1p(np.maximum(vw, 0))))
        vm = float(np.mean(vw))
        volr = float(v[j] / vm) if vm > 0 else 1.0
        vb = np.sum(v[j - 14:j])
        tbr = float(np.sum(tb[j - 14:j]) / vb) if vb > 0 else 0.5
        r1 = e / c[j - 1] - 1 if j >= 1 else 0.0
        r15 = e / c[j - 15] - 1 if j >= 15 else 0.0
        r60 = e / c[j - 60] - 1 if j >= 60 else 0.0
        ts = pd.Timestamp(t_enter // 1000, unit='s', tz='UTC')
        pos = e / lo_p if lo_p > 0 else 1.0
        lo_age = j - li

        rows.append((t_enter, ret, is_tp, t_close,
                     [volr, tbr, v240, pos, lo_age, r1, r15, r60, ts.hour, ts.dayofweek]))

    rows.sort(key=lambda x: x[0])          # 关键：按入场时间排序
    hist, heap = [], []                    # heap 装 (t_close, is_tp)
    X, E = [], []
    n_nohist = 0
    for t_enter, ret, is_tp, t_close, base in rows:
        while heap and heap[0][0] <= t_enter:
            _, ok = heapq.heappop(heap)
            hist.append(ok)
        if not hist: n_nohist += 1
        h3 = float(np.mean(hist[-3:])) if hist else -1.0
        h10 = float(np.mean(hist[-10:])) if hist else -1.0
        h3_any = 1.0 if (hist and any(hist[-3:])) else 0.0
        X.append(base + [h3, h10, h3_any, len(hist)])
        E.append((t_enter, ret, is_tp))
        heapq.heappush(heap, (t_close, is_tp))
    return np.asarray(X, dtype=float), E, n_nohist


ROWS, N_NOHIST = [], 0
for sym, (o, h, l, c, v, tb, t) in RAWS.items():
    n = len(c)
    pairs = find_lows(l, c, n)
    if not pairs: continue
    X, E, nn = build(pairs, o, h, l, c, t, v, tb)
    N_NOHIST += nn
    for (ts, ret, is_tp), feat in zip(E, X):
        ROWS.append((ts, ret, is_tp, feat))
log(f'入场点 {len(ROWS):,}  其中无历史(h3=-1) {N_NOHIST:,}  耗时 {time.time()-t0:.0f}s')

df = pd.DataFrame({'t': [r[0] for r in ROWS], 'ret': [r[1] for r in ROWS], 'tp': [r[2] for r in ROWS]})
X = np.array([r[3] for r in ROWS], dtype=float)
df['ym'] = pd.to_datetime(df['t'], unit='ms', utc=True).dt.strftime('%Y-%m')
months = sorted(df['ym'].unique())
ntr = max(1, int(len(months) * 0.75))
train_m = months[:ntr]; test_m = months[ntr:]
tr = np.where(df['ym'].isin(train_m).values)[0]
te = np.where(df['ym'].isin(test_m).values)[0]
log(f'训练 {len(tr):,}（{train_m[0]}~{train_m[-1]}）  测试 {len(te):,}（{test_m[0]}~{test_m[-1]}）')

mdl = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_depth=3,
                                     l2_regularization=1.0, min_samples_leaf=50,
                                     random_state=7)
mdl.fit(X[tr], df['tp'].values[tr])
Pte = mdl.predict_proba(X[te])[:, 1]

h3 = X[te][:, 10]; h10 = X[te][:, 11]; h3any = X[te][:, 12]; nhist = X[te][:, 13]

RULES = {
    '全部（不筛）': np.ones(len(te), dtype=bool),
    'h3_any=1（最近3次至少1次TP）': h3any >= 1.0,
    'h3>=0.33': h3 >= 0.33,
    'h3>=0.67': h3 >= 0.67,
    '有历史 n_hist>=3': nhist >= 3,
    'h3_any=1 & n_hist>=3': (h3any >= 1.0) & (nhist >= 3),
    '模型 P>=0.3': Pte >= 0.3,
    '模型 P>=0.5': Pte >= 0.5,
    'h3_any=1 & 模型P>=0.5': (h3any >= 1.0) & (Pte >= 0.5),
}


def sim(mask):
    idx = np.where(mask)[0]
    if len(idx) == 0: return None
    rs = df['ret'].values[te][idx] - FEE
    ts = df['t'].values[te][idx]
    order = np.argsort(ts)
    rs = rs[order]
    cap = START; peak = START; dd = 0.0
    for r in rs:
        cap *= (1 + POS * r)
        peak = max(peak, cap)
        dd = max(dd, (peak - cap) / peak)
    return dict(n=len(rs), mean=rs.mean() * 100, med=np.median(rs) * 100,
                win=(rs > 0).mean() * 100, tpr=df['tp'].values[te][idx].mean() * 100,
                cap=cap, dd=dd * 100)


log('')
log('=' * 100)
log('【测试段（样本外）· 前视修干净后的真实水平】')
log(f"{'规则':<32}{'n':>8}{'均值%':>9}{'中位%':>9}{'胜率%':>8}{'TP率%':>8}{'50元→':>12}{'回撤%':>8}")
for name, mask in RULES.items():
    s = sim(mask)
    if s is None:
        log(f'{name:<32}{0:>8}'); continue
    log(f"{name:<32}{s['n']:>8,}{s['mean']:>9.2f}{s['med']:>9.2f}{s['win']:>8.1f}{s['tpr']:>8.1f}{s['cap']:>12,.0f}{s['dd']:>8.1f}")

log('')
log('【h3 分档（测试段全部入场点）】')
log(f"{'h3':>8}{'n':>10}{'TP率%':>10}{'均值%':>10}")
for v in np.unique(h3):
    m = h3 == v
    if m.sum() < 30: continue
    rs = df['ret'].values[te][m]
    log(f'{v:>8.2f}{int(m.sum()):>10,}{df["tp"].values[te][m].mean()*100:>10.1f}{rs.mean()*100:>10.2f}')

log('')
log('【h3_any 二档】')
for v in [0.0, 1.0]:
    m = h3any == v
    if m.sum() == 0: continue
    rs = df['ret'].values[te][m]
    log(f'  h3_any={v:.0f}  n={int(m.sum()):>7,}  TP率 {df["tp"].values[te][m].mean()*100:>5.1f}%  均值 {rs.mean()*100:>+6.2f}%')

log('')
log('【前视审计】')
log(f'  无历史笔数 {N_NOHIST:,} / {len(ROWS):,} = {N_NOHIST/max(len(ROWS),1)*100:.1f}%')
log(f'  测试段有历史占比 {float((nhist>=1).mean())*100:.1f}%')
log(f'  测试段 n_hist 中位 {float(np.median(nhist)):.0f}')
log('  · 修法=每笔按 t_enter 排序，只有 t_close<=t_enter 的笔进 h3；不存在泄露未来的路径')

log('')
log('【判读】')
log('  · h3_any=1 的 TP率 若显著 > 全部（不筛）的 TP率 ⇒ 动量持续性真实存在')
log('  · 若 h3_any=1 与不筛几乎一样 ⇒ 旧的 85.5% 全是前视，这条线没有真实信息')
log('  · 模型 P>=0.5 若不如 h3_any=1 ⇒ 模型没有超出动量的增量')

log('')
log('=' * 100)
log('【频率：h3 各档在测试段出现多少次】')
te_t = df['t'].values[te]
days = (pd.Timestamp(int(te_t.max()) // 1000, unit='s', tz='UTC')
        - pd.Timestamp(int(te_t.min()) // 1000, unit='s', tz='UTC')).days + 1
log(f'  测试段跨度 {days} 天（{pd.Timestamp(int(te_t.min())//1000, unit="s", tz="UTC").date()} ~ {pd.Timestamp(int(te_t.max())//1000, unit="s", tz="UTC").date()}）')
m033 = (h3 > 0.1) & (h3 < 0.5)
m067 = (h3 > 0.5) & (h3 < 0.9)
m100 = h3 > 0.9
log(f"{'档位':<22}{'笔数':>8}{'次/天':>9}{'次/周':>9}{'次/年':>9}{'均值%':>9}")
for name, m in [('h3=0.33（赢1次）', m033), ('h3=0.67（赢2次）', m067),
                ('h3=1.00（赢3次）', m100), ('[0.33,0.67] 合计', m033 | m067)]:
    n = int(m.sum())
    if n == 0:
        log(f'{name:<22}{0:>8}'); continue
    mu = (df['ret'].values[te][m] - FEE).mean() * 100
    log(f'{name:<22}{n:>8,}{n/days:>9.2f}{n/days*7:>9.1f}{n/days*365:>9.0f}{mu:>9.2f}')

log('')
log('【按月分布（h3=0.33 / 0.67）】')
ym_te = pd.Series(df['ym'].values[te])
for name, m in [('h3=0.33', m033), ('h3=0.67', m067)]:
    s = ym_te[m].value_counts().sort_index()
    log(f'  {name}: ' + '  '.join(f'{k}={v}' for k, v in s.items()))

pd.DataFrame({'t': te_t, 'ret': df['ret'].values[te], 'is_tp': df['tp'].values[te],
              'h3': h3, 'h10': h10}).to_csv(
    os.path.join(OUT, 'h3_clean_rows.csv'), index=False, encoding='utf-8-sig')

out = os.path.join(OUT, 'h3_clean_summary.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
