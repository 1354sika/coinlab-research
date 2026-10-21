# -*- coding: utf-8 -*-
"""
gambler_reverse · 反做对照：同一份 σ，五种仓位规则同台
  HOLD1  满仓买入持有（最笨的赌徒，也是必须有的基准）
  HOLD0  空仓（什么都不做）
  VOL    正向  w = clip(target/σ, 0, 1)
  REV    反做  w = 1 - clip(target/σ, 0, 1)
  RAND   随机  w ~ U(0,1)
判据：
  · REV 明显 > HOLD0 且 > HOLD1  ⇒ 反着做真能赚
  · REV 只是贴着 HOLD0（≈1.0）   ⇒ 反做=空仓=没戏，不是"反向 alpha"
"""
import os, sys, json, time, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

DATA = r'D:\中信证券\coinlab\data\klines_1m_1y.csv'
OUT  = r'D:\中信证券\coinlab\evidence'

N_SYM  = int(os.environ.get('RV_SYM', '8'))
W      = int(os.environ.get('RV_W', '256'))
EPOCHS = int(os.environ.get('RV_EPOCHS', '10'))
STEPS  = int(os.environ.get('RV_STEPS', '50'))
BATCH  = int(os.environ.get('RV_BATCH', '6'))
SEG    = int(os.environ.get('RV_SEG', '1024'))
H      = int(os.environ.get('RV_H', '60'))
TARGET = float(os.environ.get('RV_TARGET', '0.0070'))
FEE    = 0.0002
BLOCK  = 1440

SYMS_ALL = ['DOGEUSDT', 'BTCUSDT', 'SOLUSDT', 'ETHUSDT',
            'TRUMPUSDT', 'XRPUSDT', 'PEPEUSDT', 'FFUSDT']
SYMS = SYMS_ALL[:N_SYM]

say = print


class Net(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(N_SYM, 32, 5, padding=2), nn.ReLU(),
            nn.Conv1d(32, 32, 5, padding=2), nn.ReLU(),
            nn.Conv1d(32, 1, 1),
        )

    def forward(self, x):
        return self.body(x).squeeze(1)


def load_panel():
    t0 = time.time()
    df = pd.read_csv(DATA, usecols=['symbol', 'open_time', 'close'], engine='c')
    df = df[df.symbol.isin(SYMS)].drop_duplicates(['symbol', 'open_time'])
    M = df.pivot(index='open_time', columns='symbol', values='close').sort_index()
    M = M.reindex(columns=SYMS).ffill().dropna()
    C = np.log(M.values.astype(np.float64))
    T, n = C.shape
    r = np.diff(C, axis=0)
    say(f'  {n} 币  {T:,} 根  [{time.time()-t0:.0f}s]')
    return r


def futures(r, h):
    """fut[i] = sum(r[i:i+h])，末尾 h 个置 NaN"""
    cs = np.cumsum(r, axis=0)
    fut = np.full_like(r, np.nan)
    fut[:-h] = cs[h:] - cs[:-h]
    return fut


def shuffle_blocks(r, rng, block=BLOCK):
    n = len(r)
    blocks = [np.arange(i, min(i + block, n)) for i in range(0, n, block)]
    rng.shuffle(blocks)
    return np.concatenate([r[b] for b in blocks])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shuffle', action='store_true')
    ap.add_argument('--tag', default='REAL')
    a = ap.parse_args()

    rng = np.random.default_rng(20261019)
    say('=' * 100)
    say(f'gambler_reverse · 反做对照 | {a.tag} | 币 {N_SYM} | W={W} H={H} target={TARGET}')
    say('=' * 100)

    r = load_panel()
    if a.shuffle:
        r = shuffle_blocks(r, rng)
        say('  [SHUF] 块序打乱已完成')

    T, N = r.shape
    cut = int(T * 0.7)
    sd = r[:cut].std()
    X = (r / sd).astype(np.float32)
    fut = futures(r, H)

    say(f'  样本 {T:,}  训练 0..{cut:,}  测试 {cut:,}..{T:,}')

    net = Net(SEG)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    say(f'  网络参数 {sum(p.numel() for p in net.parameters())/1000:.1f} K')

    # ---------- 训练：只学 logσ ----------
    for ep in range(EPOCHS):
        t0 = time.time()
        tot = 0.0
        net.train()
        for _ in range(STEPS):
            xs, ys = [], []
            for _ in range(BATCH):
                s = int(rng.integers(0, max(1, cut - SEG - H - 1)))
                xs.append(X[s:s + SEG].T)
                ys.append(fut[s:s + SEG].mean(axis=1))
            x = torch.tensor(np.stack(xs), dtype=torch.float32)
            y = torch.tensor(np.stack(ys), dtype=torch.float32)
            ls = net(x).clamp(-6.0, 2.0)
            sig = torch.exp(ls)
            mask = ~torch.isnan(y)
            yy = torch.nan_to_num(y)
            loss = 0.5 * (yy ** 2 / (sig ** 2 + 1e-12) + 2 * ls)[mask].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        say(f'  epoch {ep+1:2d}/{EPOCHS}  NLL {tot/STEPS:+.5f}  '
            f'σ均值 {float(sig.detach().mean()):.5f}  ({time.time()-t0:.0f}s)')

    # ---------- 测试段：算 σ ----------
    net.eval()
    sig_all = np.full(T, np.nan)
    with torch.no_grad():
        for s in range(cut, T - SEG, SEG // 2):
            x = torch.tensor(X[s:s + SEG].T[None], dtype=torch.float32)
            sig_all[s:s + SEG] = torch.exp(net(x).clamp(-6.0, 2.0)).numpy()

    ok = ~np.isnan(sig_all)
    say(f'  σ 有效根数 {ok.sum():,}  中位 {np.nanmedian(sig_all):.5f}')
    sig_all = pd.Series(sig_all).ffill().bfill().values

    # ---------- 五种规则 ----------
    w_vol = np.clip(TARGET / np.maximum(sig_all, 1e-9), 0, 1)
    w_rev = 1.0 - w_vol
    w_rand = rng.random(T)

    rules = {
        'HOLD1 满仓持有': np.ones(T),
        'HOLD0 空仓': np.zeros(T),
        'VOL  正向赔率': w_vol,
        'REV  反做': w_rev,
        'RAND 随机仓位': w_rand,
    }

    # 每根 w[i] 作用于 r[i+1]；换手成本 fee*|Δw|
    res = {}
    say()
    say('=' * 100)
    say('【结果】w[i] 作用于下一根收益，扣换手费 2bp')
    say('=' * 100)
    say(f'  {"规则":<16} {"均仓":>7} {"在场率":>7} {"每根净":>11} {"期末":>9} {"回撤":>8} {"四段":>28}')

    for name, wv in rules.items():
        ret = wv[:-1] * r[1:, :].mean(axis=1)   # 每根组合收益（等权币）
        d = np.r_[wv[0], np.abs(np.diff(wv))][:-1]
        ret = ret - FEE * d
        eq = np.cumprod(1.0 + ret)
        peak = np.maximum.accumulate(eq)
        dd = (eq / peak - 1.0).min()
        n = len(ret) // 4
        segs = [eq[i * n + n - 1] / (eq[i * n - 1] if i else 1.0) for i in range(4)]
        res[name] = dict(
            mean_pos=float(wv.mean()),
            inplay=float((wv > 0.05).mean()),
            per_bar=float(ret.mean()),
            final=float(eq[-1]),
            mdd=float(dd),
            segs=[float(x) for x in segs],
        )
        say(f'  {name:<16} {wv.mean():7.3f} {(wv>0.05).mean()*100:6.1f}% '
            f'{ret.mean()*100:+10.5f}% {eq[-1]:9.4f} {dd*100:7.1f}% '
            f'[{segs[0]:.3f} {segs[1]:.3f} {segs[2]:.3f} {segs[3]:.3f}]')

    # ---------- 关键诊断：w 与未来收益的相关性 ----------
    say()
    say('=' * 100)
    say('【诊断】仓位与下一根收益的相关性（正=顺向，负=反向）')
    say('=' * 100)
    rn = r[1:, :].mean(axis=1)
    for name in ['VOL  正向赔率', 'REV  反做', 'RAND 随机仓位']:
        wv = rules[name][:-1]
        c = float(np.corrcoef(wv, rn)[0, 1])
        say(f'  {name:<16} corr = {c:+.5f}')

    say()
    say('=' * 100)
    say('【判据】')
    say('=' * 100)
    rev = res['REV  反做']
    h0 = res['HOLD0 空仓']
    h1 = res['HOLD1 满仓持有']
    say(f'  反做期末 {rev["final"]:.4f}  vs  空仓 {h0["final"]:.4f}  vs  满仓 {h1["final"]:.4f}')
    if rev['final'] > h1['final'] * 1.01 and rev['final'] > 1.0:
        say('  ✅ 反做确实比满仓好且为正 ⇒ 反向有用')
    elif abs(rev['final'] - h0['final']) < 0.01:
        say('  ⚠ 反做 ≈ 空仓 ⇒ 反做不是"反向 alpha"，只是把仓位清零')
    else:
        say('  ❌ 反做没有优势')

    path = os.path.join(OUT, f'gambler_reverse_{a.tag}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    say(f'  写入 {path}')


if __name__ == '__main__':
    t = time.time()
    main()
    say(f'用时 {time.time()-t:.0f}s')
