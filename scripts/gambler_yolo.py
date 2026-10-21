# -*- coding: utf-8 -*-
"""gambler_yolo.py — 借鉴 YOLO 的单阶段「赌徒」

YOLO 里真正值钱的三件事（不是网络结构，是**问题的写法**）：
  1. **objectness ≠ 分类**：先判「这里有没有值得下注的东西」，再谈下多大。
     → 赌徒不预测涨跌，先判「这一把值不值得玩」。
  2. **一个框 = 一个事件**：一次前向输出「从哪开始、持多久、下多大」，
     而不是「每根 K 线一个仓位」再拼接。
  3. **anchor-free**：框的尺寸由模型回归，不是预设几种。→ 符合「不能用死赌法」。

本脚本一次前向输出三头：
  obj[t]  ∈ [0,1]   从第 t 根开始下注值不值
  hold[t] ∈ [1,HMAX] 值的话持多少根
  size[t] ∈ [0,1]   下多大
硬约束：
  · 不做空（仓位 ∈ [0,1]）
  · 标签只在训练期用未来；推理不用任何未来数据
  · 判据：样本外 + 打乱对照（打乱后必须变差）

用法：
  set YO_SYM=6 && python -X utf8 gambler_yolo.py
  python -X utf8 gambler_yolo.py --shuffle
"""
import os, sys, time, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA = r'D:\中信证券\coinlab\data\klines_1m_1y.csv'
OUT  = r'D:\中信证券\coinlab\evidence'

N_SYM  = int(os.environ.get('YO_SYM', '6'))
W      = int(os.environ.get('YO_W', '256'))       # 网络可视历史
HMIN   = int(os.environ.get('YO_HMIN', '120'))    # 最短持有（根）：1m 下换手太频会被手续费吃光
HMAX   = int(os.environ.get('YO_HMAX', '1440'))   # 最长持有（根）
HSTEP  = int(os.environ.get('YO_HSTEP', '4'))     # 持有长度搜索步长（标签用）
EPOCHS = int(os.environ.get('YO_EPOCHS', '6'))
STEPS  = int(os.environ.get('YO_STEPS', '80'))
BATCH  = int(os.environ.get('YO_BATCH', '8'))
SEG    = int(os.environ.get('YO_SEG', '512'))
LR     = float(os.environ.get('YO_LR', '1e-3'))
FEE    = 0.0004          # maker 0.02% x 2
TAUQ   = float(os.environ.get('YO_TAUQ', '0.6'))   # 标签阈值 = 训练段 best_ret 的该分位（自适应）
MIN_BET= 0.05            # 下注率下限（用户指定：强制）
TRAIN_FRAC = 0.70
SEED = 20261019
SHUFFLE = '--shuffle' in sys.argv

torch.manual_seed(SEED); np.random.seed(SEED)
try: torch.set_num_threads(8)
except Exception: pass


def say(*a): print(*a, flush=True)


def load_all(n_sym):
    t0 = time.time()
    df = pd.read_csv(DATA, usecols=['symbol', 'open_time', 'close', 'volume', 'taker_base'])
    df.columns = ['sym', 't', 'c', 'v', 'tb']
    df = df.dropna()
    say('  CSV 载入 %d 行  %.0fs' % (len(df), time.time() - t0))
    out = []
    for sym, g in df.groupby('sym', sort=False):
        if len(out) >= n_sym:
            break
        g = g.sort_values('t')
        out.append((sym, g['t'].values.astype(np.int64),
                    g['c'].values.astype(np.float64),
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


def targets(c, hmin, hmax, hstep, fee, train_frac, q=0.6):
    """YOLO 式标签：每根 t → (obj, hold, best_ret)。**只在训练期使用。**
       阈值 = 训练段 best_ret 的 q 分位（自适应，不写死绝对值）。"""
    n = len(c)
    best = np.full(n, -1e9, dtype=np.float32)
    arg = np.zeros(n, dtype=np.float32)
    logc = np.log(np.maximum(c, 1e-12))
    for h in range(hmin, hmax + 1, hstep):
        r = np.full(n, np.nan, dtype=np.float32)
        r[:n - h] = (logc[h:] - logc[:n - h]) - 2.0 * fee
        r = np.nan_to_num(r, nan=-1e9)
        better = r > best
        best = np.where(better, r, best)
        arg = np.where(better, h, arg)
    cut = max(1000, int(n * train_frac))
    valid = best[:cut][best[:cut] > -1e8]
    thr = float(np.quantile(valid, q)) if len(valid) > 100 else 0.0
    obj = (best > thr).astype(np.float32)
    return obj, arg, best, thr


def shuffle_blocks(X, r, block=288, rng=None):
    n = X.shape[1]; nb = n // block
    if nb < 3: return X, r
    order = rng.permutation(nb)
    idx = np.concatenate([np.arange(i * block, (i + 1) * block) for i in order])
    return X[:, idx].copy(), r[idx].copy()


class Net(nn.Module):
    """因果 1D 卷积 + 三头；输入 (B,3,L) → 输出 (B,L) 每个位置的预测。
       因果 = 左 padding 后切回 L，保证位置 t 只看 t 及之前（防前视）。"""
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv1d(3, 24, 5, padding=4)
        self.c2 = nn.Conv1d(24, 48, 5, padding=8, dilation=2)
        self.c3 = nn.Conv1d(48, 64, 5, padding=16, dilation=4)
        self.c4 = nn.Conv1d(64, 64, 5, padding=32, dilation=8)
        self.drop = nn.Dropout(0.1)
        self.h_obj = nn.Conv1d(64, 1, 1)
        self.h_hold = nn.Conv1d(64, 1, 1)
        self.h_size = nn.Conv1d(64, 1, 1)

    def forward(self, x):
        L = x.shape[2]
        h = F.relu(self.c1(x))[:, :, :L]
        h = F.relu(self.c2(h))[:, :, :L]
        h = F.relu(self.c3(h))[:, :, :L]
        h = F.relu(self.c4(h))[:, :, :L]
        h = self.drop(h)
        obj = torch.sigmoid(self.h_obj(h)).squeeze(1)
        hold = F.softplus(self.h_hold(h)).squeeze(1)
        size = torch.sigmoid(self.h_size(h)).squeeze(1)
        return obj, hold, size


def make_windows(Xseg, SEG, W):
    idx = np.arange(SEG)[:, None] + np.arange(W)[None, :]
    return np.transpose(Xseg[:, idx], (1, 0, 2)).copy()


def run_policy(obj, hold, size, r, t_open=0.5, t_exit=0.5):
    """逐根：空仓且 obj>t_open → 开 size；持仓且 (obj<t_exit 或 超过预测 hold) → 平。
       不做空。返回净收益序列（费率按换手计）。"""
    n = len(obj)
    net = np.zeros(n, dtype=np.float64)
    pos = 0.0; left = 0
    for t in range(n - 1):
        o = obj[t]
        if pos <= 0:
            if o > t_open:
                want = float(np.clip(size[t], 0.0, 1.0))
                net[t] -= FEE * want          # 开仓换手成本
                pos = want; left = int(max(HMIN, min(float(hold[t]), float(HMAX))))
        else:
            if o < t_exit or left <= 0:
                net[t] -= FEE * pos           # 平仓换手成本
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
    say('gambler_yolo · 单阶段赌徒（objectness / hold / size 三头，anchor-free，不做空）')
    say('  %s | 币 %d | W=%d H=[%d..%d] | 数据 klines_1m_1y.csv | 判据=样本外+打乱对照'
        % (tag, N_SYM, W, HMIN, HMAX))
    say('=' * 104)
    t0 = time.time()
    rng = np.random.default_rng(SEED if not SHUFFLE else SEED + 7)

    raw = load_all(N_SYM)
    data = []
    for sym, t, c, v, tb in raw:
        X, r = build_channels(c, v, tb)
        obj, hold, best, thr = targets(c, HMIN, HMAX, HSTEP, FEE, TRAIN_FRAC, TAUQ)
        if SHUFFLE:
            X, r = shuffle_blocks(X, r, 288, rng)
        data.append((sym, X, r, obj, hold, best))
        say('  %-12s %8d 根  obj 正样本率 %.1f%%  阈值 %+.4f%%  平均最优持有 %.0f 根'
            % (sym, len(r), 100 * obj.mean(), 100 * thr, hold[obj > 0].mean() if obj.sum() else 0))
    say('载入 %d 币  %.0fs' % (len(data), time.time() - t0))

    model = Net()
    say('网络参数 %.1f K（三头，anchor-free）' % (sum(p.numel() for p in model.parameters()) / 1e3))
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

    say('\n【训练】YOLO 式损失 = BCE(objectness) + 0.02×|hold| + |size|（仅正样本）')
    for ep in range(EPOCHS):
        model.train()
        lo, lh, ls, cnt = 0.0, 0.0, 0.0, 0
        for st in range(STEPS):
            xs, oys, hys, sys_ = [], [], [], []
            for _ in range(BATCH):
                k = rng.integers(0, len(data))
                _, X, r, obj, hold, best = data[k]
                tr_end = int(X.shape[1] * TRAIN_FRAC)
                if tr_end - SEG - W - 2 <= W:
                    continue
                s = int(rng.integers(W, tr_end - SEG - 2))
                xs.append(X[:, s - W:s + SEG])
                oys.append(obj[s:s + SEG]); hys.append(hold[s:s + SEG])
                sys_.append(np.clip(best[s:s + SEG] / 0.02, 0, 1))
            if not xs:
                continue
            xb = torch.from_numpy(np.stack(xs))
            op, hp, sp = model(xb)
            op = op[:, W:]; hp = hp[:, W:]; sp = sp[:, W:]
            oy = torch.from_numpy(np.stack(oys).astype(np.float32))
            hy = torch.from_numpy(np.stack(hys).astype(np.float32))
            sy = torch.from_numpy(np.stack(sys_).astype(np.float32))
            # 正负样本平衡（正样本通常 <15%）
            pos = oy.mean().clamp(min=1e-3)
            wpos = (0.5 / pos).clamp(max=20.0)
            bce = F.binary_cross_entropy(op, oy, reduction='none')
            l_obj = (bce * torch.where(oy > 0.5, wpos, torch.ones_like(oy))).mean()
            m = (oy > 0.5).float()
            denom = m.sum().clamp(min=1.0)
            l_hold = ((hp - hy).abs() * m).sum() / denom / HMAX
            l_size = ((sp - sy).abs() * m).sum() / denom
            loss = l_obj + 0.02 * l_hold + l_size
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            lo += float(l_obj); lh += float(l_hold); ls += float(l_size); cnt += 1
        say('  epoch %2d/%d  obj %.4f  hold %.4f  size %.4f  (%.0fs)'
            % (ep + 1, EPOCHS, lo / max(cnt, 1), lh / max(cnt, 1), ls / max(cnt, 1), time.time() - t0))

    say('\n' + '=' * 104)
    say('【样本外】验证段 = 各币后 %d%%，逐根前向 → 开仓/平仓，不做空' % int((1 - TRAIN_FRAC) * 100))
    model.eval()
    all_net, all_pos, per, all_true, all_hold_true, all_hold_pred = [], [], [], [], [], []
    for sym, X, r, obj, hold, best in data:
        te_s = int(X.shape[1] * TRAIN_FRAC) + W
        te_e = X.shape[1] - 2
        if te_e - te_s < 2000:
            continue
        seg_len = te_e - te_s
        outs_o, outs_h, outs_s = [], [], []
        CH = 20000
        with torch.no_grad():
            for i in range(0, seg_len, CH):
                a = te_s + i - W
                b = min(te_s + i + CH, te_e + W)
                o, h_, s_ = model(torch.from_numpy(X[:, a:b][None]))
                k = min(CH, seg_len - i)
                outs_o.append(o[0, W:W + k].numpy())
                outs_h.append(h_[0, W:W + k].numpy())
                outs_s.append(s_[0, W:W + k].numpy())
        o = np.concatenate(outs_o); h_ = np.concatenate(outs_h); s_ = np.concatenate(outs_s)
        net = run_policy(o, h_, s_, r[te_s:te_e])
        all_net.append(net)
        all_pos.append(o)
        all_true.append(obj[te_s:te_e])
        all_hold_true.append(hold[te_s:te_e])
        all_hold_pred.append(h_)
        eq = np.cumprod(1.0 + np.clip(net, -0.99, None))
        per.append((sym, len(net), float(eq[-1] - 1.0), float((o > 0.5).mean())))
    if not all_net:
        say('无验证数据'); return
    net = np.concatenate(all_net); objp = np.concatenate(all_pos)
    eq = np.cumprod(1.0 + np.clip(net, -0.99, None))
    peak = np.maximum.accumulate(eq); dd = float((eq / peak - 1).min())
    bc = block_compound(net)

    say('  %-16s %14s' % ('指标', '值'))
    say('  交易根数        %14d' % len(net))
    say('  触发率(obj>0.5) %13.1f%%' % (100 * (objp > 0.5).mean()))
    # —— 诊断：obj 头到底有没有判别力（这是「赌徒能不能避开必输局」的唯一硬指标）——
    yt = np.concatenate(all_true); hp_ = np.concatenate(all_hold_pred)
    try:
        from sklearn.metrics import roc_auc_score
        if 0 < yt.mean() < 1:
            auc = float(roc_auc_score(yt, objp))
            say('  obj AUC（样本外） %13.4f  %s' % (auc, '✅有判别力' if auc > 0.55 else ('⚠弱' if auc > 0.52 else '❌≈0.5 无预测力')))
        else:
            auc = float('nan'); say('  obj AUC               n/a（标签单一）')
    except Exception as e:
        auc = float('nan'); say('  obj AUC               失败 %s' % e)
    say('  obj 输出分布 P10/50/90  %5.3f / %5.3f / %5.3f'
        % tuple(np.percentile(objp, [10, 50, 90])))
    say('  hold 预测 P10/50/90    %5.0f / %5.0f / %5.0f 根'
        % tuple(np.percentile(hp_, [10, 50, 90])))
    say('  每根净收益均值  %+13.5f%%' % (100 * net.mean()))
    say('  期末资金        %14.4f' % eq[-1])
    say('  最大回撤        %13.1f%%' % (100 * dd))
    if len(bc) >= 2:
        tstat = bc.mean() / (bc.std(ddof=1) / np.sqrt(len(bc))) if bc.std(ddof=1) > 0 else float('nan')
        say('  周期块(n=%d) 均值 %+.3f%%  正 %d/%d  t=%+.2f'
            % (len(bc), 100 * bc.mean(), int((bc > 0).sum()), len(bc), tstat))
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
    for s in per[:4]: say('  %-12s n=%7d  收益 %+9.2f%%  触发率 %.1f%%' % (s[0], s[1], s[2], 100 * s[3]))
    say('  …')
    for s in per[-2:]: say('  %-12s n=%7d  收益 %+9.2f%%  触发率 %.1f%%' % (s[0], s[1], s[2], 100 * s[3]))
    say('  盈利币 %d/%d' % (sum(1 for s in per if s[2] > 0), len(per)))

    say('\n' + '=' * 104)
    say('【判据】%s' % tag)
    say('  ① 期末资金 > 1        : %s (%.4f)' % ('✅' if eq[-1] > 1 else '❌', eq[-1]))
    say('  ② 触发率 >= %.0f%%     : %s (%.1f%%)' % (MIN_BET * 100, '✅' if (objp > 0.5).mean() >= MIN_BET else '❌', 100 * (objp > 0.5).mean()))
    say('  ③ 最大回撤 > -30%%     : %s (%.1f%%)' % ('✅' if dd > -0.30 else '❌', 100 * dd))
    say('  ④ 四段同正            : %s' % ('✅' if ok4 else '❌'))
    if SHUFFLE:
        say('  ※ 打乱对照：必须明显差于 REAL 才算数')
    say('=' * 104)

    res = dict(tag=tag, n=int(len(net)), trig=float((objp > 0.5).mean()),
               obj_auc=auc,
               mean_net=float(net.mean()), final=float(eq[-1]), mdd=dd,
               quartiles_ok=bool(ok4),
               block_mean=float(bc.mean()) if len(bc) else None,
               block_pos=int((bc > 0).sum()) if len(bc) else 0, block_n=len(bc),
               per_sym=per)
    os.makedirs(OUT, exist_ok=True)
    fp = os.path.join(OUT, 'gambler_yolo_%s.json' % tag)
    json.dump(res, open(fp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    say('写入 %s' % fp)
    say('TOTAL %.0fs' % (time.time() - t0))


if __name__ == '__main__':
    main()
