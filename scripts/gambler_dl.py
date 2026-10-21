# -*- coding: utf-8 -*-
"""gambler_dl · 深度学习直喂序列：LSTM 对比 GBDT（AUC 0.7497）

输入：过去 60 分钟的 10 维特征序列（OHLCV 衍生，不做手工特征工程）
输出：未来 60 分钟跌 1% 概率
网络：LSTM(64) → 线性 → sigmoid；CPU 训练（无 GPU）
对照：GBDT 三模型融合 AUC(跌1%)=0.7497
"""
import os, gc, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

t0 = time.time()
D = r'D:\中信证券\coinlab\data'
OUT = r'D:\中信证券\coinlab\evidence\cycle'
SRC = os.path.join(D, 'klines_1m_502.csv')
HOLD = 60          # 未来窗口
SEQ = 60           # 过去序列长度
STEP = 200

torch.set_num_threads(max(1, os.cpu_count() - 1))

logbuf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); logbuf.append(s)

log('=' * 100)
log('gambler_dl · LSTM 直喂 60 分钟序列 → 未来 60 分钟跌 1%（对照 GBDT 0.7497）')
log('=' * 100)
log(f'threads={torch.get_num_threads()}  torch={torch.__version__}')

SEQFEAT = 10  # r1,r5,r15,r60,volr,tbr,trr,vstd,hlr,body 的每根序列
XS, YS, TS = [], [], []
for ch in pd.read_csv(SRC,
                      usecols=['symbol','open_time','close','high','low','open','volume',
                               'quote_volume','trades','taker_base'],
                      dtype={'symbol':'object','open_time':'int64','close':'float32','high':'float32',
                             'low':'float32','open':'float32','volume':'float32',
                             'quote_volume':'float32','trades':'float32','taker_base':'float32'},
                      chunksize=3_000_000):
    for sym, g in ch.groupby('symbol', sort=False):
        if len(g) < 8000:
            continue
        g = g.sort_values('open_time')
        ot = g['open_time'].values.astype('int64')
        o = g['open'].values.astype('float64')
        c = g['close'].values.astype('float64')
        h = g['high'].values.astype('float64')
        l = g['low'].values.astype('float64')
        q = g['quote_volume'].values.astype('float64')
        vol = g['volume'].values.astype('float64')
        tr = g['trades'].values.astype('float64')
        tb = g['taker_base'].values.astype('float64')
        n = len(c); m = n - HOLD
        if m <= SEQ + 200:
            continue
        cc = c[:m]
        fmin = pd.Series(l).rolling(HOLD).min().shift(-HOLD).values[:m]
        y = (fmin <= cc * 0.99).astype('int8')
        def ret(k):
            r = np.full(m, np.nan); r[k:] = cc[k:] / cc[:-k] - 1; return r
        r1, r5, r15, r60 = ret(1), ret(5), ret(15), ret(60)
        qma = pd.Series(q).rolling(60, min_periods=10).mean().values[:m]
        volr = np.where(qma > 0, q[:m] / np.maximum(qma, 1e-12), np.nan)
        tbr = np.where(vol[:m] > 0, tb[:m] / np.maximum(vol[:m], 1e-12), np.nan)
        trma = pd.Series(tr).rolling(60, min_periods=10).mean().values[:m]
        trr = np.where(trma > 0, tr[:m] / np.maximum(trma, 1e-12), np.nan)
        vstd = pd.Series(r1).rolling(15, min_periods=5).std().values
        span = np.maximum(cc, 1e-12)
        hlr = (h[:m] - l[:m]) / span
        body = np.abs(cc - o[:m]) / span
        F = np.column_stack([r1,r5,r15,r60,volr,tbr,trr,vstd,hlr,body])
        ok = np.isfinite(F).all(axis=1) & np.isfinite(fmin)
        idx = np.arange(SEQ, m, STEP)
        idx = idx[ok[idx]]
        if len(idx) == 0:
            continue
        # 序列：idx 往前 SEQ 根
        win = np.stack([F[i - SEQ:i] for i in idx])          # (n, SEQ, 10)
        win = np.nan_to_num(win, nan=0.0, posinf=5.0, neginf=-5.0)
        XS.append(win.astype('float32'))
        YS.append(y[idx]); TS.append(ot[idx])
    del ch; gc.collect()

X = np.concatenate(XS); y = np.concatenate(YS); t = np.concatenate(TS)
del XS, YS, TS; gc.collect()
log(f'样本 {len(X):,}  形状 {X.shape}  正例 {y.mean():.1%}')

# 标准化（按特征维度，用训练段统计）
cut = np.median(t)
tr = t <= cut; te = t > cut
mu = X[tr].mean(axis=(0, 1), keepdims=True)
sd = X[tr].std(axis=(0, 1), keepdims=True) + 1e-8
Xtr = np.clip((X[tr] - mu) / sd, -5, 5)
Xte = np.clip((X[te] - mu) / sd, -5, 5)
ytr = torch.from_numpy(y[tr].astype('float32'))
yte = torch.from_numpy(y[te].astype('float32'))
Xtr_t = torch.from_numpy(Xtr); Xte_t = torch.from_numpy(Xte)
log(f'训练 {tr.sum():,}  测试 {te.sum():,}（测试正例 {y[te].mean():.1%}）')

class LSTMModel(nn.Module):
    def __init__(self, nf, hid=64):
        super().__init__()
        self.lstm = nn.LSTM(nf, hid, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hid, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)

model = LSTMModel(SEQFEAT)
opt = torch.optim.Adam(model.parameters(), lr=5e-4)
lossf = nn.BCEWithLogitsLoss()
EPOCHS = 6
BS = 1024
n = len(Xtr_t)
t_fit = time.time()
for ep in range(EPOCHS):
    perm = torch.randperm(n)
    tot = 0.0
    for i in range(0, n, BS):
        b = perm[i:i+BS]
        opt.zero_grad()
        loss = lossf(model(Xtr_t[b]), ytr[b])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tot += loss.item() * len(b)
    log(f'  epoch {ep+1}/{EPOCHS} loss {tot/n:.4f}  累计 {time.time()-t_fit:.0f}s')
log(f'训练耗时 {time.time()-t_fit:.0f}s')

model.eval()
with torch.no_grad():
    outs = []
    for i in range(0, len(Xte_t), 4096):
        outs.append(torch.sigmoid(model(Xte_t[i:i+4096])).numpy())
p = np.concatenate(outs)

from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y[te], p)
log(f'\n【LSTM 测试段 AUC = {auc:.4f}】  对照 GBDT 三模型融合 = 0.7497')

yt = y[te]
N = len(yt); D1 = yt.sum()
order = np.argsort(-p); cum = np.cumsum(yt[order])
log('\n【召回率-精确率】')
log(f'{"目标召回":10s}{"阈值":8s}{"精确率":10s}')
for target in [0.25, 0.50, 0.75, 0.90]:
    k = min(max(int(np.searchsorted(cum, target * D1)) + 1, 1), N)
    thr = p[order][k - 1]
    mm = p >= thr
    log(f'{target:>8.0%}{thr:>8.2f}{yt[mm].mean():>10.1%}')

log('\n【按阈值】')
for thr in [0.5, 0.7, 0.8, 0.9]:
    mm = p >= thr
    if mm.sum() == 0:
        continue
    log(f'P>={thr:.1f}  报出 {mm.sum():>7,} ({mm.sum()/N:.1%})  精确率 {yt[mm].mean():.1%}  召回 {yt[mm].sum()/D1:.1%}')

log('\n【判读】')
log('  · LSTM 若 > 0.75 ⇒ 直喂序列有增量；< 0.75 ⇒ 与 GBDT 同一上限')
log('  · CPU 6 epoch 收敛是否充分？看最后一次 loss 是否仍在下降')

out = os.path.join(OUT, 'dl_summary.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log(f'\n报告已写 {out}')
log(f'TOTAL {time.time()-t0:.0f}s')
