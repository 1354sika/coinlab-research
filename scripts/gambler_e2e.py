# -*- coding: utf-8 -*-
"""gambler_e2e.py — 「活灵活现」首版（B 路线）

用户已确认的定义（2026-10-19）：
  · 一个赌徒，不是一群；不做空；杜绝前视
  · 输入 = 原始序列（无任何人设计的招式/特征组合）
  · 输出 = 连续仓位 [0,1]（买/卖/持由仓位变化体现，三动作）
  · 中间无人写的规则、招式列表、专家列表
  · 目标 = 周期复利（分块对数收益），强制下注率下限
  · 判据 = 只看样本外 + 打乱对照必须变差

实现要点：
  1. 只读原始列：open_time / close / volume / taker_base
  2. 归一化只用过去（rolling(...).shift(1)），无前视
  3. 训练按"连续段"采样，段内资金连续（等价真实决策链）
  4. 损失 = -段内对数收益 + 下注率下限惩罚
  5. 评估：验证段逐根前向 → 总复利 / 30 天块复利 / 分段资金 / 下注率
  6. --shuffle：块序打乱（保留块内结构，破坏跨块顺序）重跑同一流程

用法（后台）：
  set E2E_SYM=6 && set E2E_DAYS=540 && python -X utf8 gambler_e2e.py
  python -X utf8 gambler_e2e.py --shuffle
"""
import os, sys, time, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

E_DIR = r'E:\binance\klines_1m\1m'
OUT   = r'D:\中信证券\coinlab\evidence'

SYMS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT', 'DOGEUSDT',
        'ADAUSDT', 'AVAXUSDT', 'LINKUSDT', 'DOTUSDT', 'LTCUSDT', 'ATOMUSDT',
        'TRXUSDT', 'BCHUSDT', 'ETCUSDT', 'FILUSDT']

N_SYM    = int(os.environ.get('E2E_SYM', '6'))
DAYS     = int(os.environ.get('E2E_DAYS', '540'))
SEG      = int(os.environ.get('E2E_SEG', '512'))      # 训练段长度（根）
W        = int(os.environ.get('E2E_W', '256'))        # 网络可视历史（根）
EPOCHS   = int(os.environ.get('E2E_EPOCHS', '10'))
STEPS    = int(os.environ.get('E2E_STEPS', '100'))    # 每 epoch 步数
BATCH    = int(os.environ.get('E2E_BATCH', '8'))      # 每步并行段数
LR       = float(os.environ.get('E2E_LR', '1e-3'))
FEE      = 0.0004          # maker 0.02% x 2（换手）
MIN_BET  = 0.05            # 下注率下限（用户指定：强制）
COV_W    = float(os.environ.get('E2E_COV', '200'))    # 仓位-收益协方差项权重（治“仓位恒定”退化解）
SPREAD_W = float(os.environ.get('E2E_SPREAD', '0.5')) # 仓位分散度权重
BLOCK    = 43200           # 30 天 = 43200 根 1m
TRAIN_FRAC = 0.70
SEED = 20261019
SHUFFLE = '--shuffle' in sys.argv

torch.manual_seed(SEED)
np.random.seed(SEED)
try:
    torch.set_num_threads(8)
except Exception:
    pass


def say(*a):
    print(*a, flush=True)


def load_sym(sym, days):
    p = os.path.join(E_DIR, sym + '.csv')
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p, header=None, usecols=[1, 5, 6, 10],
                     names=['t', 'c', 'v', 'tb'], skiprows=1)
    df = df.dropna()
    t = df['t'].values.astype(np.int64)
    c = df['c'].values.astype(np.float64)
    v = df['v'].values.astype(np.float64)
    tb = df['tb'].values.astype(np.float64)
    if len(t) < 5000:
        return None
    if days > 0:
        m = t >= (t[-1] - days * 86400000)
        t, c, v, tb = t[m], c[m], v[m], tb[m]
    return t, c, v, tb


def build_channels(c, v, tb):
    """原始三通道，归一化只用过去（shift(1)）。无前视。"""
    n = len(c)
    r = np.zeros(n)
    r[1:] = np.log(np.maximum(c[1:], 1e-12) / np.maximum(c[:-1], 1e-12))
    s = pd.Series(r)
    mu = s.rolling(240, min_periods=60).mean().shift(1).values
    sd = s.rolling(240, min_periods=60).std().shift(1).values
    zr = np.clip((r - mu) / np.maximum(sd, 1e-8), -5, 5)

    lv = np.log1p(np.maximum(v, 0))
    lvs = pd.Series(lv)
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
    """块序打乱：保留块内结构，破坏跨块顺序。用于对照实验。"""
    n = X.shape[1]
    nb = n // block
    if nb < 3:
        return X, r
    order = rng.permutation(nb)
    idx = np.concatenate([np.arange(i * block, (i + 1) * block) for i in order])
    return X[:, idx].copy(), r[idx].copy()


class Net(nn.Module):
    """极简 1D 卷积：输入 (B,3,W) → 仓位 (B,) ∈ [0,1]。无任何手工招式。"""
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv1d(3, 16, 5, padding=2)
        self.c2 = nn.Conv1d(16, 32, 5, padding=4, dilation=2)
        self.c3 = nn.Conv1d(32, 32, 5, padding=8, dilation=4)
        self.drop = nn.Dropout(0.1)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        h = F.relu(self.c1(x))
        h = F.relu(self.c2(h))
        h = F.relu(self.c3(h))
        h = self.drop(h)
        last = h[:, :, -8:].mean(-1)
        glob = h.mean(-1)
        z = torch.cat([last, glob], dim=1)
        return torch.sigmoid(self.fc(z)).squeeze(-1)


def make_windows(Xseg, SEG, W):
    """Xseg: (3, W+SEG) → (SEG, 3, W)"""
    idx = np.arange(SEG)[:, None] + np.arange(W)[None, :]
    out = Xseg[:, idx]                       # (3, SEG, W)
    return np.transpose(out, (1, 0, 2)).copy()


def eval_segment(model, X, r, s, SEG):
    """在连续段上逐根决策，返回净收益序列（与仓位）。"""
    xs = make_windows(X[:, s - W:s + SEG], SEG, W)
    with torch.no_grad():
        w = model(torch.from_numpy(xs)).numpy().astype(np.float64)
    rn = r[s:s + SEG].astype(np.float64)
    dw = np.zeros(SEG)
    dw[1:] = w[1:] - w[:-1]
    dw[0] = w[0]
    net = w * rn - FEE * np.abs(dw)
    return net, w, rn


def block_compound(net, block=BLOCK):
    """按 30 天块算复利，返回每块收益率。"""
    nb = len(net) // block
    out = []
    for i in range(nb):
        seg = net[i * block:(i + 1) * block]
        out.append(np.expm1(np.log1p(np.clip(seg, -0.99, None)).sum()))
    return np.array(out)


def main():
    tag = 'SHUF' if SHUFFLE else 'REAL'
    say('=' * 100)
    say('gambler_e2e · 活灵活现：原始序列 → 连续仓位（无人写招式）')
    say('  %s | 币 %d | %d 天 1m | W=%d SEG=%d | 目标=周期复利(30天块) | 下注率下限 %.0f%% | 费 %.4f%%'
        % (tag, N_SYM, DAYS, W, SEG, MIN_BET * 100, FEE * 100))
    say('=' * 100)

    t0 = time.time()
    rng = np.random.default_rng(SEED if not SHUFFLE else SEED + 1)

    data = []
    used = []
    for sym in SYMS[:N_SYM]:
        o = load_sym(sym, DAYS)
        if o is None:
            say('  跳过 %s（无文件或太短）' % sym)
            continue
        t, c, v, tb = o
        X, r = build_channels(c, v, tb)
        if SHUFFLE:
            X, r = shuffle_blocks(X, r, 288, rng)
        data.append((X, r))
        used.append(sym)
        say('  %-12s %8d 根  %s ~ %s' % (
            sym, len(r),
            pd.to_datetime(t[0], unit='ms').date(), pd.to_datetime(t[-1], unit='ms').date()))
    if not data:
        say('无可用数据')
        return
    say('载入 %d 币  %.0fs' % (len(data), time.time() - t0))

    model = Net()
    n_par = sum(p.numel() for p in model.parameters())
    say('网络参数 %.1f K（无手工招式）' % (n_par / 1e3))
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

    # ---------- 训练 ----------
    say('\n【训练】每步 = %d 段 × %d 根（段内连续，等效真实决策链）' % (BATCH, SEG))
    for ep in range(EPOCHS):
        model.train()
        tot, cnt, tot_cov, tot_sd = 0.0, 0, 0.0, 0.0
        for st in range(STEPS):
            xs, rn = [], []
            for _ in range(BATCH):
                k = rng.integers(0, len(data))
                X, r = data[k]
                train_end = int(X.shape[1] * TRAIN_FRAC)
                if train_end - SEG - W - 2 <= W:
                    continue
                s = int(rng.integers(W, train_end - SEG - 2))
                xs.append(make_windows(X[:, s - W:s + SEG], SEG, W))
                rn.append(r[s:s + SEG])
            if not xs:
                continue
            xb = torch.from_numpy(np.stack(xs))
            rb = torch.from_numpy(np.stack(rn).astype(np.float32))
            B, S, C, WL = xb.shape
            w = model(xb.reshape(B * S, C, WL)).reshape(B, S)
            dw = torch.zeros_like(w)
            dw[:, 1:] = w[:, 1:] - w[:, :-1]
            dw[:, 0] = w[:, 0]
            net = w * rb - FEE * torch.abs(dw)
            logret = torch.log1p(net.clamp(min=-0.99))
            seg_ret = logret.sum(dim=1)                       # 段内对数收益（=段内复利）
            minbet_pen = torch.relu(MIN_BET - w.mean(dim=1))
            # 关键：恒定仓位时 seg_ret 对 w 的梯度 ≈ mean(r) ≈ 0，模型会停在 sigmoid(0)=0.5。
            # 协方差项 (w-ȩw)(r-ȩr) 对恒定 w 的梯度是 r-ȩr（非零），才学得动。
            wc = w - w.mean(dim=1, keepdim=True)
            rc = rb - rb.mean(dim=1, keepdim=True)
            cov = (wc * rc).mean(dim=1)
            spread = w.std(dim=1)
            loss = (-seg_ret.mean()
                    - COV_W * cov.mean()
                    + 20.0 * minbet_pen.mean()
                    - SPREAD_W * spread.mean()
                    + 1e-4 * (w ** 2).mean())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(seg_ret.mean())
            tot_cov += float(cov.mean())
            tot_sd += float(spread.mean())
            cnt += 1
        say('  epoch %2d/%d  段收益 %+.5f  协方差 %+.6f  仓位分散度 %.3f  步 %d  (%.0fs)'
            % (ep + 1, EPOCHS, tot / max(cnt, 1), tot_cov / max(cnt, 1),
               tot_sd / max(cnt, 1), cnt, time.time() - t0))

    # ---------- 评估（样本外，逐根） ----------
    say('\n' + '=' * 100)
    say('【样本外评估】验证段 = 各币后 %d%% 时间，逐根决策' % int((1 - TRAIN_FRAC) * 100))
    model.eval()
    all_net, all_w = [], []
    per_sym = []
    for k, (X, r) in enumerate(data):
        te_s = int(X.shape[1] * TRAIN_FRAC) + W
        te_e = X.shape[1] - 1
        if te_e - te_s < SEG:
            continue
        net, w, _ = eval_segment(model, X, r, te_s, te_e - te_s)
        all_net.append(net)
        all_w.append(w)
        eq = np.cumprod(1.0 + np.clip(net, -0.99, None))
        per_sym.append((used[k], len(net), float(eq[-1] - 1.0), float(w.mean())))
    if not all_w:
        say('无验证数据')
        return
    net = np.concatenate(all_net)
    w = np.concatenate(all_w)
    eq = np.cumprod(1.0 + np.clip(net, -0.99, None))
    peak = np.maximum.accumulate(eq)
    dd = float((eq / peak - 1).min())
    n = len(net)

    say('  %-14s %12s' % ('指标', '值'))
    say('  交易根数        %12d' % n)
    say('  下注率(仓>0.1)  %11.1f%%' % (100.0 * (w > 0.1).mean()))
    say('  平均仓位        %12.3f' % w.mean())
    say('  每根净收益均值  %+.5f%%' % (100 * net.mean()))
    say('  30天块复利 t 值 %+12.2f' % (
        (block_compound(net).mean() / (block_compound(net).std(ddof=1) / np.sqrt(max(len(block_compound(net)), 1))))
        if len(block_compound(net)) > 2 else float('nan')))
    say('  期末资金        %12.4f' % eq[-1])
    say('  最大回撤        %11.1f%%' % (100 * dd))

    # 四段（按时间等分，各币对齐后合并）
    q = n // 4
    say('\n【四段】(各段内资金)')
    ok4 = True
    for i in range(4):
        seg = net[i * q:(i + 1) * q] if i < 3 else net[3 * q:]
        e = float(np.exp(np.log1p(np.clip(seg, -0.99, None)).sum()))
        say('  段%d  n=%7d  段内资金 %.4f  下注率 %.1f%%' % (i + 1, len(seg), e, 100 * (w[i * q:(i + 1) * q] > 0.1).mean() if i < 3 else 100 * (w[3 * q:] > 0.1).mean()))
        ok4 &= (e > 1.0)

    bc = block_compound(net)
    if len(bc) >= 2:
        say('\n【30天块复利】n=%d  均值 %+.3f%%  中位 %+.3f%%  正块 %d/%d'
            % (len(bc), 100 * bc.mean(), 100 * np.median(bc), int((bc > 0).sum()), len(bc)))

    say('\n【分币样本外】')
    per_sym.sort(key=lambda x: -x[2])
    for s in per_sym[:5]:
        say('  %-12s n=%7d  收益 %+8.2f%%  均仓 %.3f' % s)
    say('  …')
    for s in per_sym[-3:]:
        say('  %-12s n=%7d  收益 %+8.2f%%  均仓 %.3f' % s)
    win = sum(1 for s in per_sym if s[2] > 0)
    say('  盈利币 %d/%d' % (win, len(per_sym)))

    say('\n' + '=' * 100)
    say('【判据】%s' % tag)
    say('  ① 期末资金 > 1        : %s (%.4f)' % ('✅' if eq[-1] > 1 else '❌', eq[-1]))
    say('  ② 下注率 >= %.0f%%      : %s (%.1f%%)' % (MIN_BET * 100, '✅' if (w > 0.1).mean() >= MIN_BET else '❌', 100 * (w > 0.1).mean()))
    say('  ③ 最大回撤 > -30%%     : %s (%.1f%%)' % ('✅' if dd > -0.30 else '❌', 100 * dd))
    say('  ④ 四段同正            : %s' % ('✅' if ok4 else '❌'))
    say('  ⑤ 30天块正比例 > 50%%  : %s (%d/%d)'
        % ('✅' if len(bc) and (bc > 0).mean() > 0.5 else '❌', int((bc > 0).sum()) if len(bc) else 0, len(bc)))
    if SHUFFLE:
        say('  ※ 本行是打乱对照：它必须明显差于 REAL 才算数')
    say('=' * 100)

    res = dict(tag=tag, n=n, bet_rate=float((w > 0.1).mean()), mean_w=float(w.mean()),
               mean_net=float(net.mean()), final=float(eq[-1]), mdd=dd,
               quartiles_ok=bool(ok4), block_mean=float(bc.mean()) if len(bc) else None,
               block_pos=int((bc > 0).sum()) if len(bc) else 0, block_n=len(bc),
               per_sym=[(a, b, c, d) for a, b, c, d in per_sym], syms=used)
    os.makedirs(OUT, exist_ok=True)
    fp = os.path.join(OUT, 'gambler_e2e_%s.json' % tag)
    with open(fp, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    say('写入 %s' % fp)
    say('TOTAL %.0fs' % (time.time() - t0))


if __name__ == '__main__':
    main()
