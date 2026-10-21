# -*- coding: utf-8 -*-
"""gambler_vol.py — 只用赔率：仓位只由 σ（风险）决定，不用 μ

用户裁定（2026-10-19）：μ 的 IC≈0，不要它。仓位只由实时风险(σ)决定。

数学形式（纯赔率 / 波动率目标，risk parity 的单资产版）：
    σ_t   = 模型实时估计的未来 H 根风险（单头，log σ）
    w_t   = clip( TARGET_VOL / σ_t , 0, 1 )      ← 波动低就重仓，波动高就轻仓
    收益  = w_t · r_{t+1} − c·|Δw_t|
损失只学风险，不学方向（μ 视为 0，不估计）：
    L = 0.5 · ( r²/σ² + 2·logσ )        （即 NLL(μ=0, σ)）

赌徒解释：我不猜涨跌，我只在「赔率便宜」（波动小、止损近、风险报酬比高）时下重注。

硬约束：不做空；无人工招式；因果卷积；判据=样本外+打乱对照

用法：
  set VL_SYM=8 && python -X utf8 gambler_vol.py
  python -X utf8 gambler_vol.py --shuffle
"""
import os, sys, time, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA = r'D:\中信证券\coinlab\data\klines_1m_1y.csv'
OUT  = r'D:\中信证券\coinlab\evidence'

N_SYM  = int(os.environ.get('VL_SYM', '8'))
W      = int(os.environ.get('VL_W', '256'))
HORIZ  = int(os.environ.get('VL_H', '60'))
EPOCHS = int(os.environ.get('VL_EPOCHS', '10'))
STEPS  = int(os.environ.get('VL_STEPS', '50'))
BATCH  = int(os.environ.get('VL_BATCH', '6'))
SEG    = int(os.environ.get('VL_SEG', '1024'))
LR     = float(os.environ.get('VL_LR', '1e-3'))
COST   = 0.0004
TARGET = float(os.environ.get('VL_TARGET', '0.007'))   # 目标波动（60 根 log 收益 std）
TRAIN_FRAC = 0.70
SEED = 20261019
SHUFFLE = '--shuffle' in sys.argv

torch.manual_seed(SEED); np.random.seed(SEED)
try: torch.set_num_threads(8)
except Exception: pass


def say(*a): print(*a, flush=True)


def load_all(n_sym):
    df = pd.read_csv(DATA, usecols=['symbol', 'open_time', 'close', 'volume', 'taker_base'])
    df.columns = ['sym', 't', 'c', 'v', 'tb']
    df = df.dropna()
    out = []
    for sym, g in df.groupby('sym', sort=False):
        if len(out) >= n_sym:
            break
        g = g.sort_values('t')
        out.append((sym, g['c'].values.astype(np.float64),
                    g['v'].values.astype(np.float64),
                    g['tb'].values.astype(np.float64)))
    return out


def build_channels(c, v, tb):
    n = len(c)
    r = np.zeros(n)
    r[1:] = np.log(np.maximum(c[1:], 1e-12) / np.maximum(c[:-1], 1e-12))
    s = pd.Series(r)
    mu = s.rolling(240, min_periods=60).mean().shift(1).values
    sd = s.rolling(240, min_periods=60).std().shift(1).values
    zr = np.clip((r - mu) / np.maximum(sd, 1e-8), -5, 5)
    lv = np.log1p(np.maximum(v, 0)); lvs = pd.Series(lv)
    lmu = lvs.rolling(240, min_periods=60).mean().shift(1).values
    lsd = lvs.rolling(240, min_periods=60).std().shift(1).values
    zv = np.clip((lv - lmu) / np.maximum(lsd, 1e-8), -5, 5)
    tr = pd.Series(np.clip(tb / np.maximum(v, 1e-9), 0, 1))
    tr = tr.rolling(240, min_periods=60).mean().shift(1).values
    ta = np.clip((tr - 0.5) * 4, -2, 2)
    X = np.stack([zr, zv, ta]).astype(np.float32)
    X[~np.isfinite(X)] = 0.0
    return X, r.astype(np.float32)


def fwd_ret(r, h):
    n = len(r)
    f = np.full(n, np.nan, dtype=np.float32)
    cs = np.concatenate([[0.0], np.cumsum(r.astype(np.float64))])
    f[:n - h] = (cs[h:n] - cs[:n - h]).astype(np.float32)
    return f


def shuffle_blocks(X, r, block=288, rng=None):
    n = X.shape[1]; nb = n // block
    if nb < 3: return X, r
    order = rng.permutation(nb)
    idx = np.concatenate([np.arange(i * block, (i + 1) * block) for i in order])
    return X[:, idx].copy(), r[idx].copy()


class Net(nn.Module):
    """因果 1D 卷积，单头 logσ（不估计方向）。"""
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv1d(3, 24, 5, padding=4)
        self.c2 = nn.Conv1d(24, 48, 5, padding=8, dilation=2)
        self.c3 = nn.Conv1d(48, 64, 5, padding=16, dilation=4)
        self.c4 = nn.Conv1d(64, 64, 5, padding=32, dilation=8)
        self.drop = nn.Dropout(0.1)
        self.h_ls = nn.Conv1d(64, 1, 1)

    def forward(self, x):
        L = x.shape[2]
        h = F.relu(self.c1(x))[:, :, :L]
        h = F.relu(self.c2(h))[:, :, :L]
        h = F.relu(self.c3(h))[:, :, :L]
        h = F.relu(self.c4(h))[:, :, :L]
        h = self.drop(h)
        return self.h_ls(h).squeeze(1).clamp(-9.0, 1.0)


def vol_w(ls, target=TARGET):
    """w = clip(target/σ, 0, 1)：赔率就在这个式子里（波动小 → 下重注）。"""
    return np.clip(target / np.maximum(np.exp(ls), 1e-9), 0.0, 1.0)


def block_compound(net, block=43200):
    nb = len(net) // block
    return np.array([np.expm1(np.log1p(np.clip(net[i*block:(i+1)*block], -0.99, None)).sum())
                     for i in range(nb)])


def main():
    tag = 'SHUF' if SHUFFLE else 'REAL'
    say('=' * 106)
    say('gambler_vol · 只用赔率：w = clip(target/σ, 0, 1)（不估计方向 μ）')
    say('  %s | 币 %d | W=%d SEG=%d H=%d | target_vol=%.4f | 不做空' % (tag, N_SYM, W, SEG, HORIZ, TARGET))
    say('=' * 106)
    t0 = time.time()
    rng = np.random.default_rng(SEED if not SHUFFLE else SEED + 17)

    raw = load_all(N_SYM)
    data = []
    for sym, c, v, tb in raw:
        X, r = build_channels(c, v, tb)
        if SHUFFLE:
            X, r = shuffle_blocks(X, r, 288, rng)
        y = fwd_ret(r, HORIZ)
        data.append((sym, X, r, y))
        say('  %-12s %8d 根  未来 %d 根收益 std %.4f%%  mean %+.5f%%'
            % (sym, len(r), HORIZ, 100 * np.nanstd(y), 100 * np.nanmean(y)))
    say('载入 %d 币  %.0fs' % (len(data), time.time() - t0))

    model = Net()
    say('网络参数 %.1f K（单头 logσ，不学方向）' % (sum(p.numel() for p in model.parameters()) / 1e3))
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

    say('\n【训练】只学风险：L = 0.5·(r²/σ² + 2logσ)   （μ 固定为 0）')
    for ep in range(EPOCHS):
        model.train()
        tot, sig_a, cnt = 0.0, 0.0, 0
        for st in range(STEPS):
            xs, ys = [], []
            for _ in range(BATCH):
                k = rng.integers(0, len(data))
                _, X, _, y = data[k]
                tr_end = int(X.shape[1] * TRAIN_FRAC)
                if tr_end - SEG - W - 2 <= W:
                    continue
                s = int(rng.integers(W, tr_end - SEG - 2))
                xs.append(X[:, s - W:s + SEG])
                ys.append(y[s:s + SEG])
            if not xs:
                continue
            xb = torch.from_numpy(np.stack(xs))
            ls = model(xb)[:, W:]
            yb = torch.from_numpy(np.stack(ys).astype(np.float32))
            m = torch.isfinite(yb)
            yb = torch.where(m, yb, torch.zeros_like(yb))
            nll = 0.5 * (yb ** 2 * torch.exp(-2.0 * ls) + 2.0 * ls)
            loss = (nll * m.float()).sum() / m.float().sum().clamp(min=1.0)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss)
            sig_a += float(torch.exp(ls[m]).mean()) if m.any() else 0.0
            cnt += 1
        say('  epoch %2d/%d  NLL %.5f  σ均值 %.5f  (%.0fs)'
            % (ep + 1, EPOCHS, tot / max(cnt, 1), sig_a / max(cnt, 1), time.time() - t0))

    say('\n' + '=' * 106)
    say('【样本外】w = clip(target/σ,0,1)，收益 = w·r_next − c·|Δw|')
    all_net, per, w_s, ls_s = [], [], [], []
    for sym, X, r, y in data:
        cut = int(X.shape[1] * TRAIN_FRAC)
        te_s, te_e = cut + 1, X.shape[1] - 1
        if te_e - te_s < 5000:
            continue
        ls_o = []
        CH = 20000
        with torch.no_grad():
            for i in range(0, te_e - te_s, CH):
                a = max(te_s + i - W, 0)
                b = min(te_s + i + CH, te_e + W)
                l1 = model(torch.from_numpy(X[:, a:b][None]))
                k = min(CH, te_e - (te_s + i))
                off = te_s + i - a
                ls_o.append(l1[0, off:off + k].numpy())
        ls = np.concatenate(ls_o)
        w = vol_w(ls)
        rn = r[te_s:te_e]
        dw = np.zeros(len(w)); dw[1:] = w[1:] - w[:-1]; dw[0] = w[0]
        net = w * rn - COST * np.abs(dw)
        all_net.append(net); w_s.append(w); ls_s.append(ls)
        eq = float(np.exp(np.log1p(np.clip(net, -0.99, None)).sum()))
        # 诊断：低波动 vs 高波动的未来收益是否不同（这才是本策略的 alpha 来源）
        yy = y[te_s:te_e]
        ok = np.isfinite(yy) & np.isfinite(ls)
        q30 = np.quantile(ls[ok], 0.30) if ok.sum() > 1000 else 0
        lo = yy[ok][ls[ok] <= q30].mean() if ok.sum() > 1000 else np.nan
        hi = yy[ok][ls[ok] >= np.quantile(ls[ok], 0.70)].mean() if ok.sum() > 1000 else np.nan
        per.append((sym, len(net), float(eq - 1.0), float(w.mean()), float(lo), float(hi)))
        say('  %-12s 收益 %+8.2f%%  均仓 %.4f  σ中位 %.5f  低波未来 %+.5f%%  高波未来 %+.5f%%'
            % (sym, 100 * (eq - 1), w.mean(), float(np.exp(np.median(ls))), 100 * lo, 100 * hi))
    if not all_net:
        say('无验证数据'); return
    net = np.concatenate(all_net); w = np.concatenate(w_s); ls = np.concatenate(ls_s)
    eq = np.cumprod(1.0 + np.clip(net, -0.99, None))
    peak = np.maximum.accumulate(eq); dd = float((eq / peak - 1).min())
    bc = block_compound(net)

    say('\n  %-16s %14s' % ('指标', '值'))
    say('  交易根数        %14d' % len(net))
    say('  平均仓位        %14.4f' % w.mean())
    say('  在场率(仓>5%%)   %13.1f%%' % (100 * (w > 0.05).mean()))
    say('  每根净收益均值  %+13.5f%%' % (100 * net.mean()))
    say('  期末资金        %14.4f' % eq[-1])
    say('  最大回撤        %13.1f%%' % (100 * dd))
    if len(bc) >= 2:
        tst = bc.mean() / (bc.std(ddof=1) / np.sqrt(len(bc))) if bc.std(ddof=1) > 0 else float('nan')
        say('  周期块(n=%d) 均值 %+.3f%%  正 %d/%d  t=%+.2f'
            % (len(bc), 100 * bc.mean(), int((bc > 0).sum()), len(bc), tst))
    q = len(net) // 4
    say('\n【四段】')
    ok4 = True
    for i in range(4):
        seg = net[i * q:(i + 1) * q] if i < 3 else net[3 * q:]
        e = float(np.exp(np.log1p(np.clip(seg, -0.99, None)).sum()))
        say('  段%d  n=%7d  段内资金 %.4f' % (i + 1, len(seg), e))
        ok4 &= (e > 1.0)
    say('\n【分币】')
    per.sort(key=lambda x: -x[2])
    for s in per[:4]: say('  %-12s n=%7d  收益 %+9.2f%%  均仓 %.4f  低波未来 %+.5f%%  高波未来 %+.5f%%' % s)
    say('  …')
    for s in per[-2:]: say('  %-12s n=%7d  收益 %+9.2f%%  均仓 %.4f  低波未来 %+.5f%%  高波未来 %+.5f%%' % s)
    say('  盈利币 %d/%d' % (sum(1 for s in per if s[2] > 0), len(per)))

    say('\n' + '=' * 106)
    say('【判据】%s' % tag)
    say('  ① 期末资金 > 1        : %s (%.4f)' % ('✅' if eq[-1] > 1 else '❌', eq[-1]))
    say('  ② 在场率 >= 5%%        : %s (%.1f%%)' % ('✅' if (w > 0.05).mean() >= 0.05 else '❌', 100 * (w > 0.05).mean()))
    say('  ③ 最大回撤 > -30%%     : %s (%.1f%%)' % ('✅' if dd > -0.30 else '❌', 100 * dd))
    say('  ④ 四段同正            : %s' % ('✅' if ok4 else '❌'))
    if SHUFFLE:
        say('  ※ 打乱对照：必须明显差于 REAL 才算数')
    say('=' * 106)

    res = dict(tag=tag, n=int(len(net)), mean_w=float(w.mean()), pres=float((w > 0.05).mean()),
               mean_net=float(net.mean()), final=float(eq[-1]), mdd=dd, quartiles_ok=bool(ok4),
               block_mean=float(bc.mean()) if len(bc) else None,
               block_pos=int((bc > 0).sum()) if len(bc) else 0, block_n=len(bc), per_sym=per)
    os.makedirs(OUT, exist_ok=True)
    fp = os.path.join(OUT, 'gambler_vol_%s.json' % tag)
    json.dump(res, open(fp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    say('写入 %s' % fp)
    say('TOTAL %.0fs' % (time.time() - t0))


if __name__ == '__main__':
    main()
