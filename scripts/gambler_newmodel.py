# -*- coding: utf-8 -*-
"""
gambler_newmodel · 重训新模型：三标签 × GBDT + LSTM 对照
  固定入场（A1 口径）：240min低点×1.03，当根收盘价买入
  标签 A：先触 +5% 且未先触 -6%（现用）
  标签 B：24h 内摸到 +5%（不看止损，只用来看「有无上涨潜力」）
  标签 C：24h 内最高涨幅 ≥ +3%
  模型：GBDT（快）+ LSTM（序列 60 根，只跑标签 A）
  输出：每个组合的 AUC + 精确率/召回率曲线 + 净期望
"""
import os, time, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = r'D:\中信证券\coinlab'
DATA = os.path.join(ROOT, 'data', 'klines_1m_1y.csv')
OUT = os.path.join(ROOT, 'evidence', 'cycle')

LOW_WIN = 240
CONFIRM = 0.03
HOLD = 1440
TP = 0.05
SL = -0.06
FEE = 0.005
CAP = 1500          # 每币采样上限（控制内存）
LSTM_N = 60000      # LSTM 总样本上限
SEQ = 60            # LSTM 序列长度
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]

logbuf = []
def log(s):
    print(s, flush=True)
    logbuf.append(s)


def build_features(g):
    """返回每根 bar 的特征矩阵（12 维）"""
    ot = g['open_time'].values.astype(np.int64)
    h = g['high'].values.astype(np.float64)
    l = g['low'].values.astype(np.float64)
    c = g['close'].values.astype(np.float64)
    v = g['volume'].values.astype(np.float64)
    tb = g['taker_base'].values.astype(np.float64)
    n = len(c)

    lo_win = pd.Series(l).rolling(LOW_WIN, min_periods=60).min().values
    r1 = np.zeros(n); r1[1:] = c[1:] / c[:-1] - 1
    r5 = np.zeros(n); r5[5:] = c[5:] / c[:-5] - 1
    r15 = np.zeros(n); r15[15:] = c[15:] / c[:-15] - 1
    r60 = np.zeros(n); r60[60:] = c[60:] / c[:-60] - 1
    r240 = np.zeros(n); r240[240:] = c[240:] / c[:-240] - 1
    vbase = pd.Series(v).rolling(288, min_periods=60).mean().values
    volr = np.where(vbase > 0, v / np.maximum(vbase, 1e-12), np.nan)
    tbr = np.where(v > 0, tb / np.maximum(v, 1e-12), np.nan)
    rets = r1[1:]
    vstd = pd.Series(rets).rolling(60, min_periods=30).std().values
    vstd = np.concatenate([[np.nan], vstd])
    vstd1440 = pd.Series(rets).rolling(1440, min_periods=240).std().values
    vstd1440 = np.concatenate([[np.nan], vstd1440])
    hi1440 = pd.Series(h).rolling(1440, min_periods=60).max().values
    pos = np.where(hi1440 > 0, c / hi1440, np.nan)
    hour = ((ot // 3_600_000) % 24).astype(np.float64)
    dow = ((ot // 86_400_000) % 7).astype(np.float64)

    F = np.column_stack([r1, r5, r15, r60, r240, volr, tbr,
                         vstd, vstd1440, hour, dow, pos])
    return F, lo_win, c, h, l, ot


def main():
    t0 = time.time()
    log('=' * 100)
    log('gambler_newmodel · 三标签 × GBDT/LSTM 对照')
    log('=' * 100)
    log(f'  入场 240min低点×{1+CONFIRM:.2f}  当根收盘价买入  CAP={CAP}/币')
    log(f'  标签 A=先触+5%且未先触-6%  B=24h摸到+5%  C=24h最高>=3%')
    log(f'  LSTM: 序列 {SEQ} 根，总样本上限 {LSTM_N:,}')

    df = pd.read_csv(DATA)
    syms = sorted(df['symbol'].unique())
    log(f'币 {len(syms)}  行 {len(df):,}')

    # 三标签共用同一批触发点
    Xs, ysA, ysB, ysC, ts, seqs = [], [], [], [], [], []

    for si, sym in enumerate(syms):
        g = df[df['symbol'] == sym].sort_values('open_time')
        if len(g) < LOW_WIN + HOLD + 10:
            continue
        F, lo_win, c, h, l, ot = build_features(g)
        n = len(c)
        entry = lo_win * (1 + CONFIRM)
        o = g['open'].values.astype(np.float64)
        m = (c >= entry) & (o <= entry * 1.03)
        m[0] = False
        m = m & np.isfinite(lo_win) & np.isfinite(F[:, 11]) & np.isfinite(F[:, 8])
        idxs = np.where(m)[0]
        if len(idxs) == 0:
            continue
        if len(idxs) > CAP:
            idxs = idxs[::max(1, len(idxs) // CAP)][:CAP]

        feats = F[idxs]
        ok = np.isfinite(feats).all(axis=1)
        IDX = idxs[ok]
        Fs = feats[ok]
        if len(IDX) == 0:
            continue

        px = c[IDX]
        tp_p = px * (1 + TP)
        sl_p = px * (1 - abs(SL))
        yA = np.zeros(len(IDX), dtype=np.int8)
        yB = np.zeros(len(IDX), dtype=np.int8)
        yC = np.zeros(len(IDX), dtype=np.int8)
        for k in range(len(IDX)):
            idx = IDX[k]
            end = min(idx + HOLD, n - 1)
            if end <= idx + 1:
                continue
            seg_h = h[idx + 1:end + 1]
            seg_l = l[idx + 1:end + 1]
            hit_tp = np.where(seg_h >= tp_p[k])[0]
            hit_sl = np.where(seg_l <= sl_p[k])[0]
            i_tp = int(hit_tp[0]) if len(hit_tp) else 10 ** 9
            i_sl = int(hit_sl[0]) if len(hit_sl) else 10 ** 9
            if i_tp < 10 ** 9:
                yB[k] = 1
            if i_tp < 10 ** 9 and i_tp < i_sl:
                yA[k] = 1
            if len(seg_h) and (seg_h.max() / px[k] - 1.0) >= 0.03:
                yC[k] = 1

        Xs.append(Fs); ysA.append(yA); ysB.append(yB); ysC.append(yC)
        ts.append(ot[IDX])

        # 为 LSTM 保存序列（每个触发点前 SEQ 根 bar 的 12 维特征）
        if len(seqs) * 1 < LSTM_N:
            need = min(len(IDX), max(1, (LSTM_N - sum(len(s) for s in seqs)) // 1))
            pick = IDX[::max(1, len(IDX) // max(1, need))][:need]
            for idx in pick:
                if idx < SEQ:
                    continue
                seg = F[idx - SEQ:idx]
                if np.isfinite(seg).all():
                    seqs.append((seg, int(yA[list(IDX).index(idx)] if idx in IDX else 0)))

        if (si + 1) % 20 == 0:
            log(f'  {sym}  {si+1}/{len(syms)}  {time.time()-t0:.0f}s')

    X = np.vstack(Xs)
    yA = np.concatenate(ysA); yB = np.concatenate(ysB); yC = np.concatenate(ysC)
    tt = np.concatenate(ts)
    order = np.argsort(tt)
    X = X[order]; yA = yA[order]; yB = yB[order]; yC = yC[order]

    n = len(X)
    tr = int(n * 0.7)
    log(f'\n总触发点 {n:,}  训练 {tr:,}  测试 {n-tr:,}')
    log(f'  标签A 基线 {yA.mean():.1%}   标签B {yB.mean():.1%}   标签C {yC.mean():.1%}')

    # ---------- GBDT 三标签 ----------
    log('\n' + '=' * 100)
    log('【一】GBDT · 三标签对照')
    log('=' * 100)
    for name, y in [('A 先触+5%', yA), ('B 摸到+5%', yB), ('C 24h最高>=3%', yC)]:
        mdl = HistGradientBoostingClassifier(max_iter=300, max_depth=6,
                                             learning_rate=0.08, random_state=0)
        mdl.fit(X[:tr], y[:tr])
        p = mdl.predict_proba(X[tr:])[:, 1]
        auc = roc_auc_score(y[tr:], p)
        npos = int(y[tr:].sum())
        log(f'\n--- 标签 {name}  测试 {len(y)-tr:,}  正样本 {npos:,} ({y[tr:].mean():.1%})  AUC={auc:.4f} ---')
        log(f'    {"阈值":>5} {"报出":>7} {"精确率":>7} {"召回率":>7} {"净期望":>9}')
        p_star = (abs(SL) + FEE) / (TP + abs(SL))
        for lo in THRESHOLDS:
            mm = p >= lo
            if mm.sum() < 30:
                continue
            prec = y[tr:][mm].mean()
            rec = int(y[tr:][mm].sum()) / max(1, npos)
            ne = prec * TP - (1 - prec) * abs(SL) - FEE
            log(f'    {lo:>5.1f} {int(mm.sum()):>7,} {prec:>6.1%} {rec:>6.1%} {ne*100:>+8.2f}%')
        log(f'    （平衡线 {p_star:.1%}）')

    # ---------- LSTM（标签 A）----------
    log('\n' + '=' * 100)
    log('【二】LSTM · 标签 A（序列输入）')
    log('=' * 100)
    try:
        import torch
        import torch.nn as nn
        torch.manual_seed(0)
        np.random.seed(0)

        arr = np.stack([s for s, _ in seqs]).astype(np.float32)
        lab = np.array([l for _, l in seqs], dtype=np.float32)
        log(f'  LSTM 样本 {len(arr):,}  序列 {arr.shape[1]} × {arr.shape[2]}  dim  正样本 {lab.mean():.1%}')

        # 标准化
        mu = arr.mean(axis=(0, 1), keepdims=True)
        sd = arr.std(axis=(0, 1), keepdims=True) + 1e-8
        arr = (arr - mu) / sd

        nt = int(len(arr) * 0.7)
        Xtr_t = torch.from_numpy(arr[:nt]); ytr_t = torch.from_numpy(lab[:nt]).unsqueeze(1)
        Xte_t = torch.from_numpy(arr[nt:]); yte_t = lab[nt:]

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(input_size=arr.shape[2], hidden_size=64,
                                    num_layers=2, batch_first=True, dropout=0.2)
                self.head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(),
                                          nn.Dropout(0.2), nn.Linear(32, 1))
            def forward(self, x):
                out, _ = self.lstm(x)
                return self.head(out[:, -1, :])

        net = Net()
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        lossf = nn.BCEWithLogitsLoss()
        bs = 256
        EPOCHS = 8
        for ep in range(EPOCHS):
            net.train()
            perm = torch.randperm(nt)
            tot = 0.0
            for i in range(0, nt, bs):
                bi = perm[i:i + bs]
                opt.zero_grad()
                out = net(Xtr_t[bi])
                loss = lossf(out, ytr_t[bi])
                loss.backward()
                opt.step()
                tot += float(loss)
            if (ep + 1) % 2 == 0:
                log(f'    epoch {ep+1}/{EPOCHS}  loss {tot/(nt/bs):.4f}  ({time.time()-t0:.0f}s)')

        net.eval()
        with torch.no_grad():
            p_lstm = torch.sigmoid(net(Xte_t)).numpy().ravel()
        auc_l = roc_auc_score(yte_t, p_lstm)
        npos_l = int(yte_t.sum())
        p_star = (abs(SL) + FEE) / (TP + abs(SL))
        log(f'\n  LSTM 测试 {len(yte_t):,}  正样本 {npos_l:,} ({yte_t.mean():.1%})  AUC={auc_l:.4f}')
        log(f'    {"阈值":>5} {"报出":>7} {"精确率":>7} {"召回率":>7} {"净期望":>9}')
        for lo in THRESHOLDS:
            mm = p_lstm >= lo
            if mm.sum() < 30:
                continue
            prec = yte_t[mm].mean()
            rec = int(yte_t[mm].sum()) / max(1, npos_l)
            ne = prec * TP - (1 - prec) * abs(SL) - FEE
            log(f'    {lo:>5.1f} {int(mm.sum()):>7,} {prec:>6.1%} {rec:>6.1%} {ne*100:>+8.2f}%')
        log(f'    （平衡线 {p_star:.1%}）')
    except Exception as e:
        log(f'  LSTM 失败：{type(e).__name__}: {e}')

    log('\n' + '=' * 100)
    log('【判读】')
    log('  · 比 AUC：哪个模型/标签最高，就是最值得继续做的')
    log('  · 比「AUC 与平衡线的距离」：AUC 高但不超平衡线 = 还是亏')
    log('  · 召回率 5% 以上且净期望 > 0 才算真正可用')
    log('=' * 100)

    out = os.path.join(OUT, 'newmodel_summary.txt')
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(logbuf))
    log(f'\n报告已写 {out}')
    log(f'TOTAL {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
