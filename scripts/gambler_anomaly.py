# -*- coding: utf-8 -*-
"""gambler_anomaly.py — objectness = 「当下状态异常」（不预测未来）

用户裁定（2026-10-19）：YOLO 的 objectness 检测「当下就在输入里的东西」，
而不是「只存在于未来的机会」。所以本版彻底去掉未来标签：

  · 自编码器只在**训练段**学「正常序列长什么样」（因果卷积，无前视）
  · 每根的重构误差 = 异常分 = objectness（**模型自己定义什么叫异常**）
  · 交易规则完全不用未来：
        异常分（相对历史分位）> 阈值 → 开仓，仓位 = 归一化异常度
        异常分回落到常态 → 平仓
  · 不做空；判据仍是样本外 + 打乱对照

这与前三版的根本区别：**不看方向、不猜收益、不设「跌 5%」这类招式**。
赌徒只做一件事：牌桌异动时下注，异动散了就走。

用法：
  set AN_SYM=8 && python -X utf8 gambler_anomaly.py
  python -X utf8 gambler_anomaly.py --shuffle
"""
import os, sys, time, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA = r'D:\中信证券\coinlab\data\klines_1m_1y.csv'
OUT  = r'D:\中信证券\coinlab\evidence'

N_SYM   = int(os.environ.get('AN_SYM', '8'))
W       = int(os.environ.get('AN_W', '512'))
EPOCHS  = int(os.environ.get('AN_EPOCHS', '8'))
STEPS   = int(os.environ.get('AN_STEPS', '40'))
BATCH   = int(os.environ.get('AN_BATCH', '6'))
SEG     = int(os.environ.get('AN_SEG', '1024'))
LR      = float(os.environ.get('AN_LR', '1e-3'))
FEE     = 0.0004
HMIN    = int(os.environ.get('AN_HMIN', '60'))
HMAX    = int(os.environ.get('AN_HMAX', '1440'))
OPEN_Q  = float(os.environ.get('AN_OPENQ', '0.97'))   # 异常分历史分位 > 此值才开仓
EXIT_Q  = float(os.environ.get('AN_EXITQ', '0.70'))   # 回落到此分位以下平仓
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


def shuffle_blocks(X, r, block=288, rng=None):
    n = X.shape[1]; nb = n // block
    if nb < 3: return X, r
    order = rng.permutation(nb)
    idx = np.concatenate([np.arange(i * block, (i + 1) * block) for i in order])
    return X[:, idx].copy(), r[idx].copy()


class AE(nn.Module):
    """因果 1D 卷积自编码器：输入 (B,3,L) → 重构 (B,3,L)。无任何未来信息。"""
    def __init__(self):
        super().__init__()
        self.e1 = nn.Conv1d(3, 16, 5, padding=4)
        self.e2 = nn.Conv1d(16, 32, 5, padding=8, dilation=2)
        self.e3 = nn.Conv1d(32, 24, 5, padding=16, dilation=4)
        self.d1 = nn.Conv1d(24, 24, 5, padding=16, dilation=4)
        self.d2 = nn.Conv1d(24, 16, 5, padding=8, dilation=2)
        self.d3 = nn.Conv1d(16, 3, 5, padding=4)

    def forward(self, x):
        L = x.shape[2]
        h = F.relu(self.e1(x))[:, :, :L]
        h = F.relu(self.e2(h))[:, :, :L]
        z = F.relu(self.e3(h))[:, :, :L]
        h = F.relu(self.d1(z))[:, :, :L]
        h = F.relu(self.d2(h))[:, :, :L]
        y = self.d3(h)[:, :, :L]
        return y


def anomaly_scores(model, X, bs=200000):
    """逐根重构误差（三通道 MSE），因果卷积保证位置 t 只用 t 之前。"""
    model.eval()
    L = X.shape[1]
    out = np.empty(L, dtype=np.float32)
    with torch.no_grad():
        for s in range(0, L, bs):
            e = min(s + bs, L)
            xb = torch.from_numpy(X[:, s:e][None])
            y = model(xb)[0].numpy()
            out[s:e] = ((X[:, s:e] - y) ** 2).mean(axis=0)
    return out


def run_policy(score, r, thr_open, thr_exit, hmin, hmax):
    """完全不用未来的交易规则：异常分高 → 开仓，仓位=归一化异常度；回落 → 平仓。"""
    n = len(score)
    net = np.zeros(n)
    pos = 0.0; left = 0
    span = max(thr_open - thr_exit, 1e-9)
    for t in range(n - 1):
        s = score[t]
        if pos <= 0:
            if s > thr_open:
                want = float(np.clip((s - thr_open) / span + 0.5, 0.1, 1.0))
                net[t] -= FEE * want
                pos = want; left = hmax
        else:
            if s < thr_exit or left <= 0:
                net[t] -= FEE * pos
                pos = 0.0; left = 0
        net[t] += pos * r[t + 1]
        if pos > 0:
            left -= 1
    return net


def block_compound(net, block=43200):
    nb = len(net) // block
    return np.array([np.expm1(np.log1p(np.clip(net[i*block:(i+1)*block], -0.99, None)).sum())
                     for i in range(nb)])


def main():
    tag = 'SHUF' if SHUFFLE else 'REAL'
    say('=' * 104)
    say('gambler_anomaly · objectness = 当下状态异常（自编码器重构误差，无未来标签）')
    say('  %s | 币 %d | W=%d SEG=%d | 开仓分位 %.2f 平仓分位 %.2f | 不做空'
        % (tag, N_SYM, W, SEG, OPEN_Q, EXIT_Q))
    say('=' * 104)
    t0 = time.time()
    rng = np.random.default_rng(SEED if not SHUFFLE else SEED + 11)

    raw = load_all(N_SYM)
    data = []
    for sym, c, v, tb in raw:
        X, r = build_channels(c, v, tb)
        if SHUFFLE:
            X, r = shuffle_blocks(X, r, 288, rng)
        data.append((sym, X, r))
        say('  %-12s %8d 根' % (sym, len(r)))
    say('载入 %d 币  %.0fs' % (len(data), time.time() - t0))

    model = AE()
    say('自编码器参数 %.1f K（无监督，只用训练段学「正常」）'
        % (sum(p.numel() for p in model.parameters()) / 1e3))
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-6)

    say('\n【训练】MSE 重构误差，只用各币前 %d%% 时间' % int(TRAIN_FRAC * 100))
    for ep in range(EPOCHS):
        model.train()
        tot, cnt = 0.0, 0
        for st in range(STEPS):
            xs = []
            for _ in range(BATCH):
                k = rng.integers(0, len(data))
                _, X, _ = data[k]
                tr_end = int(X.shape[1] * TRAIN_FRAC)
                if tr_end - W - 2 <= 0:
                    continue
                s = int(rng.integers(0, tr_end - W - 2))
                xs.append(X[:, s:s + W])
            if not xs:
                continue
            xb = torch.from_numpy(np.stack(xs))
            y = model(xb)
            loss = F.mse_loss(y, xb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss); cnt += 1
        say('  epoch %2d/%d  MSE %.5f  (%.0fs)' % (ep + 1, EPOCHS, tot / max(cnt, 1), time.time() - t0))

    say('\n' + '=' * 104)
    say('【样本外】异常分阈值 = 训练段分位（无前视）')
    all_net, per, trig_all, sp_all = [], [], [], []
    for sym, X, r in data:
        sc = anomaly_scores(model, X)
        cut = int(X.shape[1] * TRAIN_FRAC)
        tr_sc = sc[W:cut]
        thr_open = float(np.quantile(tr_sc, OPEN_Q))
        thr_exit = float(np.quantile(tr_sc, EXIT_Q))
        te_s, te_e = cut + 1, X.shape[1] - 1
        if te_e - te_s < 5000:
            continue
        net = run_policy(sc[te_s:te_e], r[te_s:te_e], thr_open, thr_exit, HMIN, HMAX)
        all_net.append(net)
        sp_all.append(sc[te_s:te_e])
        eq = np.cumprod(1.0 + np.clip(net, -0.99, None))
        per.append((sym, len(net), float(eq[-1] - 1.0), float((net != 0).mean())))
        say('  %-12s 阈值 %.5f/%.5f  测试 %d 根  收益 %+7.2f%%  在场率 %.1f%%'
            % (sym, thr_open, thr_exit, len(net), 100 * (eq[-1] - 1), 100 * (net != 0).mean()))
    if not all_net:
        say('无验证数据'); return
    net = np.concatenate(all_net)
    eq = np.cumprod(1.0 + np.clip(net, -0.99, None))
    peak = np.maximum.accumulate(eq); dd = float((eq / peak - 1).min())
    bc = block_compound(net)

    say('\n  %-16s %14s' % ('指标', '值'))
    say('  交易根数        %14d' % len(net))
    say('  在场率(净收益≠0) %13.1f%%' % (100 * (net != 0).mean()))
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
    for s in per[:4]: say('  %-12s n=%7d  收益 %+9.2f%%  在场率 %.1f%%' % s)
    say('  …')
    for s in per[-2:]: say('  %-12s n=%7d  收益 %+9.2f%%  在场率 %.1f%%' % s)
    say('  盈利币 %d/%d' % (sum(1 for s in per if s[2] > 0), len(per)))

    say('\n' + '=' * 104)
    say('【判据】%s' % tag)
    say('  ① 期末资金 > 1        : %s (%.4f)' % ('✅' if eq[-1] > 1 else '❌', eq[-1]))
    say('  ② 在场率 >= 5%%        : %s (%.1f%%)' % ('✅' if (net != 0).mean() >= 0.05 else '❌', 100 * (net != 0).mean()))
    say('  ③ 最大回撤 > -30%%     : %s (%.1f%%)' % ('✅' if dd > -0.30 else '❌', 100 * dd))
    say('  ④ 四段同正            : %s' % ('✅' if ok4 else '❌'))
    if SHUFFLE:
        say('  ※ 打乱对照：必须明显差于 REAL 才算数')
    say('=' * 104)

    res = dict(tag=tag, n=int(len(net)), pres=float((net != 0).mean()),
               mean_net=float(net.mean()), final=float(eq[-1]), mdd=dd,
               quartiles_ok=bool(ok4),
               block_mean=float(bc.mean()) if len(bc) else None,
               block_pos=int((bc > 0).sum()) if len(bc) else 0, block_n=len(bc),
               per_sym=per)
    os.makedirs(OUT, exist_ok=True)
    fp = os.path.join(OUT, 'gambler_anomaly_%s.json' % tag)
    json.dump(res, open(fp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    say('写入 %s' % fp)
    say('TOTAL %.0fs' % (time.time() - t0))


if __name__ == '__main__':
    main()
