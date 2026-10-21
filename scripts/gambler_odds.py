# -*- coding: utf-8 -*-
"""gambler_odds.py — 预测 + 赔率一起搞（用户 2026-10-19 裁定）

前三版的问题（用户指出）：全都在做预测（v1/v2 猜涨跌、v3 猜异常），
**从来没有把「赔率」作为模型的输出。**

本版：一条网络两个头，联合学习收益的**整个分布**：
    μ   = 预期收益（弱预测，模型自己决定多弱）
    logσ= 对数风险（实时赔率的分母）
损失用高斯负对数似然（学分布，不是学点）：
    L = (r - μ)² / (2σ²) + logσ
仓位由凯利给出（不是拍的，是算出来的）：
    w = clip( (μ - c) / σ² , 0, 1 ) × KELLY
    c = 往返成本；μ 小或 σ 大时 w 自动→0，不需要硬阈值
硬约束：不做空；无人工招式；因果卷积防前视；判据=样本外+打乱对照

用法：
  set OD_SYM=8 && python -X utf8 gambler_odds.py
  python -X utf8 gambler_odds.py --shuffle
"""
import os, sys, time, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA = r'D:\中信证券\coinlab\data\klines_1m_1y.csv'
OUT  = r'D:\中信证券\coinlab\evidence'

N_SYM  = int(os.environ.get('OD_SYM', '8'))
W      = int(os.environ.get('OD_W', '256'))
HORIZ  = int(os.environ.get('OD_H', '60'))        # 预测/持有视野（根）
EPOCHS = int(os.environ.get('OD_EPOCHS', '10'))
STEPS  = int(os.environ.get('OD_STEPS', '50'))
BATCH  = int(os.environ.get('OD_BATCH', '6'))
SEG    = int(os.environ.get('OD_SEG', '1024'))
LR     = float(os.environ.get('OD_LR', '1e-3'))
COST   = 0.0004          # 往返成本（maker 0.02% x 2）
KELLY  = float(os.environ.get('OD_K', '0.25'))    # 四分之一凯利（防爆）
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
    """未来 h 根的累计对数收益（标签，只在训练期用）"""
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
    """因果 1D 卷积 + 双头（μ, logσ）。输入 (B,3,L) → (B,L)。"""
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv1d(3, 24, 5, padding=4)
        self.c2 = nn.Conv1d(24, 48, 5, padding=8, dilation=2)
        self.c3 = nn.Conv1d(48, 64, 5, padding=16, dilation=4)
        self.c4 = nn.Conv1d(64, 64, 5, padding=32, dilation=8)
        self.drop = nn.Dropout(0.1)
        self.h_mu = nn.Conv1d(64, 1, 1)
        self.h_ls = nn.Conv1d(64, 1, 1)

    def forward(self, x):
        L = x.shape[2]
        h = F.relu(self.c1(x))[:, :, :L]
        h = F.relu(self.c2(h))[:, :, :L]
        h = F.relu(self.c3(h))[:, :, :L]
        h = F.relu(self.c4(h))[:, :, :L]
        h = self.drop(h)
        mu = self.h_mu(h).squeeze(1)
        ls = self.h_ls(h).squeeze(1).clamp(-9.0, 1.0)
        return mu, ls


def kelly_w(mu, ls, cost=COST, k=KELLY):
    """w = clip((μ-c)/σ², 0, 1)·k —— 赔率就在这个式子里。"""
    sig2 = np.exp(2.0 * ls)
    return np.clip((mu - cost) / np.maximum(sig2, 1e-12) * k, 0.0, 1.0)


def block_compound(net, block=43200):
    nb = len(net) // block
    return np.array([np.expm1(np.log1p(np.clip(net[i*block:(i+1)*block], -0.99, None)).sum())
                     for i in range(nb)])


def main():
    tag = 'SHUF' if SHUFFLE else 'REAL'
    say('=' * 106)
    say('gambler_odds · 预测(μ) + 赔率(σ) 合起来定仓位：w = clip((μ-c)/σ²,0,1)·k')
    say('  %s | 币 %d | W=%d SEG=%d H=%d | k=%.2f c=%.4f | 不做空' % (tag, N_SYM, W, SEG, HORIZ, KELLY, COST))
    say('=' * 106)
    t0 = time.time()
    rng = np.random.default_rng(SEED if not SHUFFLE else SEED + 13)

    raw = load_all(N_SYM)
    data = []
    for sym, c, v, tb in raw:
        X, r = build_channels(c, v, tb)
        y = fwd_ret(r, HORIZ)
        if SHUFFLE:
            X, r = shuffle_blocks(X, r, 288, rng)
            y = fwd_ret(r, HORIZ)
        data.append((sym, X, r, y))
        say('  %-12s %8d 根  未来 %d 根收益 std %.4f%%  mean %+.5f%%'
            % (sym, len(r), HORIZ, 100 * np.nanstd(y), 100 * np.nanmean(y)))
    say('载入 %d 币  %.0fs' % (len(data), time.time() - t0))

    model = Net()
    say('网络参数 %.1f K（双头：μ / logσ）' % (sum(p.numel() for p in model.parameters()) / 1e3))
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

    say('\n【训练】高斯 NLL：L = (r-μ)²/(2σ²) + logσ（学分布，不是学点）')
    for ep in range(EPOCHS):
        model.train()
        tot, mu_a, sig_a, cnt = 0.0, 0.0, 0.0, 0
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
            mu_p, ls_p = model(xb)
            mu_p = mu_p[:, W:]; ls_p = ls_p[:, W:]
            yb = torch.from_numpy(np.stack(ys).astype(np.float32))
            m = torch.isfinite(yb)
            yb = torch.where(m, yb, torch.zeros_like(yb))
            sig2 = torch.exp(2.0 * ls_p)
            nll = 0.5 * ((yb - mu_p) ** 2 / sig2 + 2.0 * ls_p)
            loss = (nll * m.float()).sum() / m.float().sum().clamp(min=1.0)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss)
            mu_a += float(mu_p[m].mean()) if m.any() else 0.0
            sig_a += float(torch.exp(ls_p[m]).mean()) if m.any() else 0.0
            cnt += 1
        say('  epoch %2d/%d  NLL %.5f  μ均值 %+.5f  σ均值 %.5f  (%.0fs)'
            % (ep + 1, EPOCHS, tot / max(cnt, 1), mu_a / max(cnt, 1), sig_a / max(cnt, 1), time.time() - t0))

    say('\n' + '=' * 106)
    say('【样本外】逐根：w = clip((μ-c)/σ²,0,1)·k，收益 = w·r_next − c·|Δw|')
    all_net, per, mu_s, w_s, ic_list = [], [], [], [], []
    for sym, X, r, y in data:
        cut = int(X.shape[1] * TRAIN_FRAC)
        te_s, te_e = cut + 1, X.shape[1] - 1
        if te_e - te_s < 5000:
            continue
        mu_o, ls_o = [], []
        CH = 20000
        with torch.no_grad():
            for i in range(0, te_e - te_s, CH):
                a = te_s + i - W
                b = min(te_s + i + CH, te_e + W)
                a = max(a, 0)
                m1, l1 = model(torch.from_numpy(X[:, a:b][None]))
                k = min(CH, te_e - (te_s + i))
                off = te_s + i - a
                mu_o.append(m1[0, off:off + k].numpy())
                ls_o.append(l1[0, off:off + k].numpy())
        mu = np.concatenate(mu_o); ls = np.concatenate(ls_o)
        w = kelly_w(mu, ls)
        rn = r[te_s:te_e]
        dw = np.zeros(len(w)); dw[1:] = w[1:] - w[:-1]; dw[0] = w[0]
        net = w * rn - COST * np.abs(dw)
        all_net.append(net); mu_s.append(mu); w_s.append(w)
        # 用未来 H 根真实收益算 IC（诊断：μ 到底有没有信息）
        yy = fwd_ret(r, HORIZ)[te_s:te_e]
        ok = np.isfinite(yy)
        ic = float(np.corrcoef(mu[ok], yy[ok])[0, 1]) if ok.sum() > 1000 else float('nan')
        ic_list.append(ic)
        per.append((sym, len(net), float(np.exp(np.log1p(np.clip(net, -0.99, None)).sum()) - 1.0),
                    float(w.mean()), ic))
        say('  %-12s 收益 %+8.2f%%  均仓 %.4f  μ均值 %+.5f  IC %+.4f'
            % (sym, 100 * (np.exp(np.log1p(np.clip(net, -0.99, None)).sum()) - 1),
               100 * 0 + w.mean(), mu.mean(), ic))
    if not all_net:
        say('无验证数据'); return
    net = np.concatenate(all_net); w = np.concatenate(w_s); mu = np.concatenate(mu_s)
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
    say('  μ 均值/P10/90   %+.5f  %+.5f  %+.5f' % (mu.mean(), np.percentile(mu, 10), np.percentile(mu, 90)))
    say('  IC(μ vs 未来收益) %+13.4f   逐币均值 %+.4f' % (
        float(np.nanmean(ic_list)), float(np.nanmean(ic_list))))
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
    for s in per[:4]: say('  %-12s n=%7d  收益 %+9.2f%%  均仓 %.4f  IC %+.4f' % s)
    say('  …')
    for s in per[-2:]: say('  %-12s n=%7d  收益 %+9.2f%%  均仓 %.4f  IC %+.4f' % s)
    say('  盈利币 %d/%d' % (sum(1 for s in per if s[2] > 0), len(per)))

    say('\n' + '=' * 106)
    say('【判据】%s' % tag)
    say('  ① 期末资金 > 1        : %s (%.4f)' % ('✅' if eq[-1] > 1 else '❌', eq[-1]))
    say('  ② 在场率 >= 5%%        : %s (%.1f%%)' % ('✅' if (w > 0.05).mean() >= 0.05 else '❌', 100 * (w > 0.05).mean()))
    say('  ③ 最大回撤 > -30%%     : %s (%.1f%%)' % ('✅' if dd > -0.30 else '❌', 100 * dd))
    say('  ④ 四段同正            : %s' % ('✅' if ok4 else '❌'))
    say('  ⑤ IC(μ) > 0.01        : %s (%+.4f)' % ('✅' if np.nanmean(ic_list) > 0.01 else '❌', float(np.nanmean(ic_list))))
    if SHUFFLE:
        say('  ※ 打乱对照：必须明显差于 REAL 才算数')
    say('=' * 106)

    res = dict(tag=tag, n=int(len(net)), mean_w=float(w.mean()), pres=float((w > 0.05).mean()),
               mean_net=float(net.mean()), final=float(eq[-1]), mdd=dd, quartiles_ok=bool(ok4),
               mean_mu=float(mu.mean()), ic=float(np.nanmean(ic_list)),
               block_mean=float(bc.mean()) if len(bc) else None,
               block_pos=int((bc > 0).sum()) if len(bc) else 0, block_n=len(bc), per_sym=per)
    os.makedirs(OUT, exist_ok=True)
    fp = os.path.join(OUT, 'gambler_odds_%s.json' % tag)
    json.dump(res, open(fp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    say('写入 %s' % fp)
    say('TOTAL %.0fs' % (time.time() - t0))


if __name__ == '__main__':
    main()
