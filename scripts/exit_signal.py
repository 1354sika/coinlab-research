# -*- coding: utf-8 -*-
"""出场模型固化：训练一次并保存，之后可对任意时点给出「还会不会创新高」的概率。

用法：
  python -X utf8 exit_signal.py train                 # 训练并保存模型
  python -X utf8 exit_signal.py signal BTCUSDT        # 用最新 5m K 线给当前信号
  python -X utf8 exit_signal.py signal BTCUSDT 2.1    # 指定当前浮盈 %（默认按 24 根前入场算）

输出：
  P(未来 8 小时内还会创出更高点) 与建议（>=θ 留，<θ 走），θ 由训练段数据决定
"""
import os
import sys
import json
import time
import urllib.request
import requests
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = 'D:/中信证券/coinlab/data/klines_5m_3y.csv'
MODEL = os.path.join(HERE, '..', 'models', 'exit_q2.pkl')
META = os.path.join(HERE, '..', 'models', 'exit_q2.json')
COST = 0.0004
H = 96
W = 24
PER = 300
THETA_DEFAULT = 0.80
API = 'https://data-api.binance.vision/api/v3/klines'


def feats(ret, volr, g, hold):
    """ret/volr: 最近 W 根（已按时间升序）；g/hold: 标量或数组"""
    n = len(g)
    wr = np.repeat(ret[None, :], n, axis=0)
    wv = np.repeat(volr[None, :], n, axis=0)
    return np.concatenate([wr, wv, np.asarray(g).reshape(-1, 1), np.asarray(hold).reshape(-1, 1)], axis=1)


def build(df):
    XS, YS, FC = [], [], []
    rng = np.random.default_rng(17)
    for sym, gg in df.groupby('symbol'):
        c = gg['close'].to_numpy(float); v = gg['volume'].to_numpy(float)
        n = len(c)
        if n < H + W + 10:
            continue
        ret = np.zeros(n); ret[1:] = c[1:] / c[:-1] - 1
        vma = pd.Series(v).rolling(24).mean().to_numpy()
        volr = v / vma
        idx = np.arange(W + 2, n - H - 1, 3)
        if len(idx) > PER:
            idx = rng.choice(idx, size=PER, replace=False); idx.sort()
        jj = np.arange(H + 1)
        p0 = c[idx][:, None]
        fc = c[idx[:, None] + jj] / p0 - 1
        pos = idx[:, None] + np.arange(H + 1)[None, :]
        # 每个出场候选时点 j=1..H
        G = fc[:, 1:H]
        suf = np.maximum.accumulate(fc[:, ::-1], axis=1)[:, ::-1]
        Q = suf[:, 2:H + 1]
        Y = (Q > G + COST).astype(int)
        WR = np.stack([ret[pos[:, 1:H] - w] for w in range(W, 0, -1)], axis=-1)
        WV = np.stack([volr[pos[:, 1:H] - w] for w in range(W, 0, -1)], axis=-1)
        HOLD = np.tile(np.arange(1, H, dtype=float), (len(idx), 1))
        X = np.concatenate([WR.reshape(-1, W), WV.reshape(-1, W),
                            G.reshape(-1, 1), HOLD.reshape(-1, 1)], axis=1)
        XS.append(X.astype(np.float32)); YS.append(Y.ravel()); FC.append(fc[:, 1:H])
    X = np.concatenate(XS); Y = np.concatenate(YS); FC = np.concatenate(FC)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, Y, FC


def train():
    t0 = time.time()
    df = pd.read_csv(DATA, usecols=['symbol', 'open_time', 'close', 'volume'])
    df = df.sort_values(['symbol', 'open_time'])
    X, Y, FC = build(df)
    n = len(FC)
    split = int(n * 0.8)
    J = H - 1
    Xtr = X[:split * J]; Ytr = Y[:split * J]
    Xva = X[split * J:]; Yva = Y[split * J:]
    mdl = HistGradientBoostingClassifier(max_iter=120, max_depth=6, learning_rate=0.08, random_state=0)
    mdl.fit(Xtr, Ytr)
    # θ：在验证段用净收益选
    pv = mdl.predict_proba(Xva)[:, 1].reshape(-1, J)
    best_th, best_pnl = THETA_DEFAULT, -9
    Fva = FC[split:]
    for th in np.arange(0.10, 0.91, 0.05):
        take = pv < th
        has = take.any(1)
        jex = np.where(has, take.argmax(1), J - 1)
        pnl = Fva[np.arange(len(Fva)), jex] - COST
        if has.mean() < 0.05:
            continue
        if pnl.mean() > best_pnl:
            best_pnl, best_th = pnl.mean(), float(th)
    os.makedirs(os.path.dirname(MODEL), exist_ok=True)
    import joblib
    joblib.dump(mdl, MODEL)
    json.dump({'theta': best_th, 'val_pnl': best_pnl, 'H': H, 'W': W, 'cost': COST,
               'trained_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'paths': int(n)},
              open(META, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print('训练完成 %.0fs  θ=%.2f  验证段每笔 %.3f%%  路径 %d' % (time.time() - t0, best_th, best_pnl * 100, n))
    print('已保存 ' + MODEL)


def signal(sym, g_pct=None):
    import joblib
    if not os.path.exists(MODEL):
        sys.exit('模型未训练：先跑 train')
    mdl = joblib.load(MODEL)
    meta = json.load(open(META, encoding='utf-8'))
    url = API + '?symbol=%s&interval=5m&limit=%d' % (sym, H + W + 5)
    b = requests.get(API, params={'symbol': sym, 'interval': '5m', 'limit': H + W + 5}, timeout=20).json()
    c = np.array([float(x[4]) for x in b])
    v = np.array([float(x[5]) for x in b])
    ret = np.zeros(len(c)); ret[1:] = c[1:] / c[:-1] - 1
    vma = pd.Series(v).rolling(24).mean().to_numpy()
    volr = v / vma
    # 当前时点 = 最后一根；假设入场价给出，或按 H 根前
    if g_pct is None:
        hold = 24
        entry_px = c[-1 - hold]
        g = c[-1] / entry_px - 1
    else:
        entry_px = c[-1] / (1 + g_pct / 100.0)
        g = g_pct / 100.0
        hold = 24
    xr = ret[-W:]; xv = volr[-W:]
    X = feats(xr, xv, [g], [hold])
    X = np.nan_to_num(X)
    p = float(mdl.predict_proba(X)[0, 1])
    theta = meta['theta']
    verdict = '留（还有新高概率）' if p >= theta else '走（概率低于阈值）'
    print('%s  当前 %.6f  浮盈 %.2f%%  持有 %d 根' % (sym, c[-1], g * 100, hold))
    print('P(未来 %.0f 小时内创新高) = %.3f   θ=%.2f   => %s' % (H * 5 / 60, p, theta, verdict))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit('用法: train | signal <SYMBOL> [浮盈%%]')
    if sys.argv[1] == 'train':
        train()
    else:
        signal(sys.argv[2] if len(sys.argv) > 2 else 'BTCUSDT',
               float(sys.argv[3]) if len(sys.argv) > 3 else None)
