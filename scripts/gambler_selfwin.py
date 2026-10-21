"""
gambler_selfwin.py · 端到端赌徒 v1

模型自己决定三件事（训练后可直接看到）：
  1. 往回看多久        → 4 个尺度的注意力权重（15/60/180/480 根）
  2. 用什么特征        → 卷积核自己学
  3. 买 / 持 / 卖      → 3 类输出，每根 K 线都给

标签（未来 60 根内最先进哪个）：
  先触 +2% → 买(1)
  先触 -1% → 卖(2)
  都没触   → 持(0)

时间切分，无前视。
"""
import sys, io, os, time, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
np.seterr(all='ignore')
torch.set_num_threads(6)

DATA = r'D:\中信证券\coinlab\data\klines_1m_1y.csv'
OUT  = r'D:\中信证券\coinlab\evidence\cycle'

LOOKBACK = 480          # 最长看 8 小时；模型自己决定用其中多长
FUTURE   = 60           # 标签看未来 1 小时
TP, SL   = 0.02, 0.01
N_SYM    = 30           # 币数
N_SAMP   = 4000         # 每币采样点数
SEED     = 7

log = lambda s='': print(s, flush=True)


class SelfWindowNet(nn.Module):
    """4 个尺度的并行卷积 + 可学习注意力权重（= 模型自己选窗口）"""
    def __init__(self, ch=7, scales=(15, 60, 180, 480), hid=48):
        super().__init__()
        self.scales = scales
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(ch, hid, kernel_size=k, padding=k // 2),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            ) for k in scales
        ])
        # 模型自己学「哪个尺度重要」——训练完这组权重就是它选的窗口
        self.scale_logit = nn.Parameter(torch.zeros(len(scales)))
        self.head = nn.Sequential(
            nn.Linear(hid, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 3),
        )

    def forward(self, x):
        # x: (B, ch, T)
        feats = torch.stack([c(x).squeeze(-1) for c in self.convs], dim=1)  # (B, S, hid)
        w = F.softmax(self.scale_logit, dim=0)                              # (S,)
        pooled = (feats * w.view(1, -1, 1)).sum(1)                          # (B, hid)
        return self.head(pooled), w


def main():
    t0 = time.time()
    log('=' * 90)
    log('gambler_selfwin · 端到端赌徒 v1（模型自己决定看多久）')
    log('=' * 90)
    log(f'  最长回看 {LOOKBACK} 根 / 标签看未来 {FUTURE} 根 / TP {TP:.0%} SL {SL:.0%}')
    log(f'  尺度 {15,60,180,480} 根 → 模型自己加权')

    log('\n读数据...')
    dt = {'symbol': 'category', 'open_time': 'int64', 'open': 'float32', 'high': 'float32',
          'low': 'float32', 'close': 'float32', 'volume': 'float32',
          'taker_base': 'float32', 'taker_quote': 'float32'}
    df = pd.read_csv(DATA, usecols=list(dt.keys()), dtype=dt)
    syms = sorted(df.symbol.unique())[:N_SYM]
    log(f'  {len(df):,} 行，用前 {len(syms)} 币')

    rng = np.random.default_rng(SEED)
    X, Y, TS = [], [], []
    stat = {'buy': 0, 'sell': 0, 'hold': 0}

    for si, sym in enumerate(syms):
        g = df[df.symbol == sym].sort_values('open_time')
        n = len(g)
        if n < LOOKBACK + FUTURE + 100:
            continue
        o = g.open.values; h = g.high.values; l = g.low.values
        c = g.close.values; v = g.volume.values
        tb = g.taker_base.values; tq = g.taker_quote.values
        t = g.open_time.values

        cand = np.arange(LOOKBACK, n - FUTURE - 1)
        if len(cand) > N_SAMP:
            cand = rng.choice(cand, N_SAMP, replace=False)

        for i in cand:
            # 7 通道，做相对化
            base = c[i]
            if not np.isfinite(base) or base <= 0:
                continue
            sl = slice(i - LOOKBACK, i)
            ch = np.stack([
                (o[sl] / base - 1), (h[sl] / base - 1), (l[sl] / base - 1),
                (c[sl] / base - 1),
                np.log1p(v[sl] * 1000),
                tb[sl] / np.maximum(tb[sl] + tq[sl], 1e-9) - 0.5,   # 主动买占比(去中心)
                np.log1p(tq[sl] * 1000),
            ]).astype(np.float32)
            if not np.isfinite(ch).all():
                continue

            # 标签
            fut_h = h[i+1:i+1+FUTURE]; fut_l = l[i+1:i+1+FUTURE]
            up = fut_h >= base * (1 + TP)
            dn = fut_l <= base * (1 - SL)
            iu = np.argmax(up) if up.any() else 10**9
            idn = np.argmax(dn) if dn.any() else 10**9
            if iu < idn:
                y = 1
            elif idn < iu:
                y = 2
            else:
                y = 0
            stat[['hold', 'buy', 'sell'][y]] += 1
            X.append(ch); Y.append(y); TS.append(int(t[i]))

        if (si + 1) % 5 == 0:
            log(f'  {si+1}/{len(syms)}  样本 {len(X):,}  {time.time()-t0:.0f}s')

    X = np.stack(X); Y = np.array(Y, dtype=np.int64); TS = np.array(TS)
    log(f'\n样本 {len(X):,}  分布 {stat}')

    # 时间切分
    order = np.argsort(TS)
    cut = int(len(order) * 0.7)
    tr, te = order[:cut], order[cut:]
    Xtr, Ytr = torch.from_numpy(X[tr]), torch.from_numpy(Y[tr])
    Xte, Yte = torch.from_numpy(X[te]), torch.from_numpy(Y[te])
    log(f'训练 {len(tr):,}  测试 {len(te):,}')

    net = SelfWindowNet()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    # 类别不均衡：持 是多数，加权让 买/卖 不吃亏
    cnt = np.bincount(Ytr.numpy(), minlength=3).astype(np.float32)
    wt = torch.tensor((cnt.sum() / (3 * np.maximum(cnt, 1))), dtype=torch.float32)
    log(f'类别权重 {wt.numpy().round(2)}')

    log('\n训练...')
    B = 256
    for ep in range(6):
        perm = torch.randperm(len(Xtr))
        tot = 0.0
        for k in range(0, len(perm), B):
            idx = perm[k:k+B]
            logits, _ = net(Xtr[idx])
            loss = F.cross_entropy(logits, Ytr[idx], weight=wt)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss)
        log(f'  epoch {ep+1}/6  loss {tot/(len(perm)/B):.4f}  {time.time()-t0:.0f}s')

    # 评估
    net.eval()
    with torch.no_grad():
        lg, w = net(Xte)
        pr = torch.softmax(lg, 1).numpy()
        w = w.numpy()
    pred = pr.argmax(1)
    acc = (pred == Yte.numpy()).mean()
    log(f'\n测试段 准确率 {acc:.3f}  基线(全猜持) {(Yte.numpy()==0).mean():.3f}')

    # 混淆
    log('\n混淆矩阵（行=真 列=预测）  0持 1买 2卖')
    cm = np.zeros((3, 3), int)
    for a, b in zip(Yte.numpy(), pred):
        cm[a, b] += 1
    for i in range(3):
        log(f'  {["持","买","卖"][i]}  {cm[i]}')

    # 模型自己选的窗口
    log('\n【模型自己选的窗口】')
    for k, kk in enumerate((15, 60, 180, 480)):
        log(f'  {kk:>3} 根 ({kk}分钟)  权重 {w[k]:.3f}  {"█" * int(w[k]*60)}')

    # 买/卖信号的精确率
    log('\n【按预测档】')
    for cls, name in [(1, '买'), (2, '卖')]:
        m = pred == cls
        if m.sum() == 0:
            log(f'  {name}: 0 笔')
            continue
        hit = (Yte.numpy()[m] == cls).mean()
        log(f'  {name}: 报出 {m.sum():,}  真为{name} {hit:.1%}')

    # 存
    np.save(os.path.join(OUT, 'selfwin_w.npy'), w)
    rep = os.path.join(OUT, 'selfwin_summary.txt')
    with open(rep, 'w', encoding='utf-8') as f:
        f.write(f'样本 {len(X):,}  训练 {len(tr):,} 测试 {len(te):,}\n')
        f.write(f'测试准确率 {acc:.3f}  基线 {(Yte.numpy()==0).mean():.3f}\n')
        f.write(f'模型自选窗口权重 {dict(zip((15,60,180,480), w.round(3).tolist()))}\n')
    log(f'\n报告 {rep}')
    log(f'TOTAL {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
