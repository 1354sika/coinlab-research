# -*- coding: utf-8 -*-
r"""
gambler_cross.py — 截面维度：状态里加「相对强弱」

sizing 的结论：单笔优势 0.85% 太小，数学天花板约 1%。
要突破，必须让「相似状态」更有信息量。

截面思路：同币历史重复不是唯一信息源，**币群结构重复**可能更强。
  · 状态 8 维：r5, r15, r60, volr, sig  +  rel_r5, rel_r60, rank_r5
    - rel_r5 / rel_r60 = 该币的 r5/r60 减去同一时刻 60 币的均值
    - rank_r5 = 该币 r5 在同一时刻 60 币里的排名（0~1）
  · K=5 局部记忆（不变）
  · 真实约束（每币独立子账户、同币不重叠、连亏停牌）

对比：
  A 同币记忆（5 维）= full2 基线
  B 截面增强（8 维）
两者用同一套切分/约束，只差状态维度。
"""
import sys, io, os, time
from collections import deque
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
np.seterr(all='ignore')

DATA = r'D:\中信证券\coinlab\data\klines_5m_3y.csv'
OUT  = r'D:\中信证券\coinlab\evidence\cross.txt'
NSYM = int(os.environ.get('CX_NSYM', '60'))

COST      = 0.0009
H         = 176
STEP_TR   = 100
STEP_TE   = 50
OBSERVE   = 0.50
NBLOCK    = 30
K         = 5
KELLY     = 0.25
POS_MAX   = 0.02
STOP_N    = 8
STOP_LOSS = 0.004
STOP_BARS = 50
H_STEPS   = int(np.ceil(H/STEP_TE))
BENCH     = 1.2845

t0 = time.time()
def say(*a): print(*a, flush=True)

def rmean(x, k):
    c = np.concatenate([[0.0], np.cumsum(x)])
    out = np.full(len(x), np.nan)
    out[k-1:] = (c[k:] - c[:-k]) / k
    return out

def feats(g):
    c = g['close'].values.astype(np.float64)
    v = g['volume'].values.astype(np.float64)
    r5 = np.concatenate([np.full(5, np.nan), c[5:]/c[:-5]-1.0])
    r15 = np.concatenate([np.full(15, np.nan), c[15:]/c[:-15]-1.0])
    r60 = np.concatenate([np.full(60, np.nan), c[60:]/c[:-60]-1.0])
    vma = rmean(v, 60)
    volr = v/np.where(vma > 0, vma, np.nan)
    sig = pd.Series(c).pct_change().rolling(240).std().values
    fwd = np.full(len(c), np.nan)
    fwd[:-H] = c[H:]/c[:-H] - 1.0 - COST
    return r5, r15, r60, volr, sig, fwd

def run_one(blocks, cols, need_cross):
    """blocks: 该币按时间排序的 DataFrame（含特征列 + fwd + open_time）"""
    F = blocks[cols].values.astype(np.float64)
    y = blocks['fwd'].values.astype(np.float64)
    n = len(F)
    if n < 2000: return None
    ok = np.isfinite(F).all(axis=1) & np.isfinite(y)
    F, y = F[ok], y[ok]
    n = len(F)
    if n < 1000: return None
    cut0 = int(n*OBSERVE)
    if cut0 < 300: return None
    mu = F[:cut0].mean(0); sd = F[:cut0].std(0)+1e-9
    Fz = (F-mu)/sd
    edges = np.linspace(cut0, n, NBLOCK+1).astype(int)
    e_l, v_l, r_l = [], [], []
    for i in range(NBLOCK):
        tr_end, te_s, te_e = edges[i], edges[i], edges[i+1]
        if te_e <= te_s: continue
        sub = np.arange(0, tr_end, STEP_TR)
        if len(sub) < 50: continue
        nn = NearestNeighbors(n_neighbors=min(K, len(sub)), algorithm='kd_tree').fit(Fz[sub])
        te_idx = np.arange(te_s, te_e, STEP_TE)
        _, nb = nn.kneighbors(Fz[te_idx])
        r = y[sub][nb]
        e_l.append(r.mean(1)); v_l.append(r.var(1)+1e-10); r_l.append(y[te_idx])
    if not e_l: return None
    exp = np.concatenate(e_l); var = np.concatenate(v_l); R = np.concatenate(r_l)
    raw = np.where(exp > 0, np.clip(exp/np.maximum(var, 1e-10)*KELLY, 0, POS_MAX), 0.0)
    pause = 0; gh = deque(maxlen=40); P = np.zeros_like(raw); nxt = 0; nt = 0
    for k in range(len(raw)):
        if k < nxt: continue
        if pause > 0:
            pause -= 1; nxt = k+1; continue
        if raw[k] <= 0: continue
        P[k] = raw[k]; nxt = k+H_STEPS; nt += 1
        gh.append(raw[k]*R[k])
        if len(gh) >= STOP_N and sum(list(gh)[-STOP_N:]) < -STOP_LOSS:
            pause = STOP_BARS
    pnl = P*R
    cap = np.cumprod(1.0+pnl)
    dd = (cap/np.maximum.accumulate(cap)-1.0).min()
    n4 = len(pnl); segs = []
    for i in range(4):
        aa = i*n4//4; bb = (i+1)*n4//4
        segs.append(np.cumprod(1.0+pnl[aa:bb])[-1] if bb > aa else 1.0)
    wavg = (P*R).sum()/max(P.sum(), 1e-9)
    return dict(cap=cap[-1], dd=dd, segs=segs, n_trades=nt, wavg=wavg, pmean=P.mean())

def main():
    say('='*100)
    say('gambler_cross · 截面维度（相对强弱）能不能把单笔优势做大')
    say('='*100)
    say(f'K={K} 不重叠{H_STEPS}步 | 凯利×{KELLY} 上限{POS_MAX*100:.0f}% | 基准 {BENCH:.4f}')
    say('='*100)

    df = pd.read_csv(DATA, usecols=['symbol','open_time','open','high','low','close','volume'])
    df['symbol'] = df['symbol'].astype('category')
    keep = df['symbol'].value_counts().head(NSYM).index
    df = df[df['symbol'].isin(keep)].copy()
    df = df.sort_values(['symbol','open_time']).reset_index(drop=True)
    say(f'  载入 {len(df):,} 行 / {df.symbol.nunique()} 币  {time.time()-t0:.0f}s')

    # 按币算原始特征
    parts = []
    for sym, g in df.groupby('symbol', observed=True):
        if len(g) < 5000: continue
        r5, r15, r60, volr, sig, fwd = feats(g)
        parts.append(pd.DataFrame(dict(symbol=str(sym), open_time=g['open_time'].values,
                                       r5=r5, r15=r15, r60=r60, volr=volr, sig=sig, fwd=fwd)))
    A = pd.concat(parts, ignore_index=True)
    say(f'  特征算完 {len(A):,} 行  {time.time()-t0:.0f}s')

    # 截面：同一 open_time 的市场均值 + 排名
    g_ = A.groupby('open_time', sort=False)
    A['mkt_r5']  = g_['r5'].transform('mean')
    A['mkt_r60'] = g_['r60'].transform('mean')
    A['rel_r5']  = A['r5'] - A['mkt_r5']
    A['rel_r60'] = A['r60'] - A['mkt_r60']
    A['rank_r5'] = g_['r5'].transform(lambda s: s.rank(pct=True))
    say(f'  截面算完 {time.time()-t0:.0f}s')

    cols_base  = ['r5','r15','r60','volr','sig']
    cols_cross = cols_base + ['rel_r5','rel_r60','rank_r5']

    say()
    say('='*100)
    say('【一】两种状态维度对比（真实约束，60 币等权）')
    say('='*100)
    res = {}
    for tag, cols in (('A 同币记忆(5维)', cols_base), ('B 截面增强(8维)', cols_cross)):
        finals, dds, segs_sum, trades, wavgs, pm = [], [], np.zeros(4), 0, [], []
        cnt = 0
        for sym, g in A.groupby('symbol', observed=True):
            if len(g) < 5000: continue
            r = run_one(g, cols, tag.startswith('B'))
            if r is None: continue
            finals.append(r['cap']); dds.append(r['dd']); segs_sum += np.array(r['segs'])
            trades += r['n_trades']; wavgs.append(r['wavg']); pm.append(r['pmean']); cnt += 1
        if cnt == 0: continue
        total = float(np.mean(finals)); dd = float(np.mean(dds)); seg_avg = segs_sum/cnt
        ok4 = bool(all(v > 1.0 for v in seg_avg))
        res[tag] = dict(total=total, dd=dd, segs=seg_avg, ok4=ok4, trades=trades,
                        wavg=float(np.mean(wavgs)), pmean=float(np.mean(pm)), cnt=cnt)
        say(f'  {tag:<16} 子账户{cnt:>3}  期末 {total:>8.4f}  收益 {(total-1)*100:>+7.2f}%  回撤 {dd*100:>7.2f}%')
        say(f'  {"":<16} 单笔 {res[tag]["wavg"]*100:>+7.3f}%  均仓 {res[tag]["pmean"]*100:>5.2f}%  笔数 {trades:>6}  四段{"✅" if ok4 else "❌"} [{seg_avg[0]:.3f} {seg_avg[1]:.3f} {seg_avg[2]:.3f} {seg_avg[3]:.3f}]')
        say(f'  {"":<16} {time.time()-t0:.0f}s')

    say()
    say('='*100)
    say('【二】判据')
    say('='*100)
    if len(res) == 2:
        a = res['A 同币记忆(5维)']; b = res['B 截面增强(8维)']
        say(f'  单笔优势：A {a["wavg"]*100:+.3f}%  →  B {b["wavg"]*100:+.3f}%   （{(b["wavg"]-a["wavg"])*100:+.3f}pp）')
        say(f'  总资产：  A {a["total"]:.4f}  →  B {b["total"]:.4f}')
        say(f'  回撤：    A {a["dd"]*100:.2f}%  →  B {b["dd"]*100:.2f}%')
        better = b['wavg'] > a['wavg']
        say(f'  {"✅ 截面确实提高了单笔优势" if better else "❌ 截面没有提高单笔优势"}')
        good = b['total'] > 1.0 and b['dd'] > -0.30 and b['ok4']
        say(f'  B 是否达标（>1、回撤<30%、四段全赚）：{"✅" if good else "❌"}')

    say(f'\n写入 {OUT}  ·  TOTAL {time.time()-t0:.0f}s')
    with open(OUT, 'w', encoding='utf-8') as f:
        for k, v in res.items():
            f.write(f'{k}\t{v["total"]:.4f}\t{v["dd"]:.4f}\t{v["wavg"]:.6f}\n')

if __name__ == '__main__':
    main()
