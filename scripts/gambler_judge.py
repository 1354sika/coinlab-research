# -*- coding: utf-8 -*-
"""gambler 守门员：唯一判据入口（四段同号 + 每段赢 SHUF）

为什么要有这个文件：
  之前所有版本都用"全局平均"判生死，结果被某一段大赚掩盖另外三段
  （v3 只做多：全局 +0.0406% 看着合格，拆开是 +0.044 / −0.028 / +0.128 / +0.019）。
  从今天起，**任何打法想留下来，只能走这个判据**，不许再看全局平均。

两条判据：
  ① 四段同号：样本外按时间四分，四段单笔均值必须全部 > 0
  ② 每段赢 SHUF：每段"真牌均值 − 打乱牌均值"必须 > 0
     （SHUF = 价格不动、只打乱特征与未来的对齐，代表"一副没信息的假牌"）

任一条不过 ⇒ alive=False，并给出死于哪一条。
"""
import numpy as np


def split4(t, ret, t_ref=None):
    """按时间四分位切段，返回 4 段 (n, mean, t_stat)"""
    t = np.asarray(t, dtype=np.float64)
    ret = np.asarray(ret, dtype=np.float64)
    if len(ret) < 40:
        return []
    q = np.quantile(t, [0, .25, .5, .75, 1.0])
    out = []
    for i in range(4):
        lo, hi = q[i], q[i + 1]
        m = (t >= lo) & (t <= hi) if i == 3 else (t >= lo) & (t < hi)
        a = ret[m]
        if len(a) < 10:
            out.append(dict(n=len(a), mean=np.nan, t=np.nan, ok=False))
            continue
        tst = a.mean() / (a.std(ddof=1) / np.sqrt(len(a)))
        out.append(dict(n=len(a), mean=float(a.mean()), t=float(tst), ok=True))
    return out


def judge(real_t, real_ret, shuf_t=None, shuf_ret=None):
    """唯一判据：返回 (alive, reason, detail)

    real_t / real_ret：真牌每笔的时间戳与净收益
    shuf_t / shuf_ret：打乱牌的同口径数据；不传则第 ② 条自动不通过
                （不允许"没对照就算过"）
    """
    segs = split4(real_t, real_ret)
    detail = dict(segs=segs)

    if len(segs) != 4 or not all(s['ok'] for s in segs):
        return False, '样本不足以切四段', detail

    # ① 四段同号
    bad = [i + 1 for i, s in enumerate(segs) if not (s['mean'] > 0)]
    if bad:
        return False, f'四段不同号（第 {bad} 段 ≤ 0）', detail

    # ② 每段赢 SHUF
    if shuf_t is None or shuf_ret is None or len(shuf_ret) == 0:
        return False, '缺 SHUF 对照（不许无对照通过）', detail
    ssegs = split4(shuf_t, shuf_ret)
    if len(ssegs) != 4:
        return False, 'SHUF 样本不足以切四段', detail
    lose = [i + 1 for i in range(4)
            if not (segs[i]['mean'] > ssegs[i]['mean'])]
    detail['shuf_segs'] = ssegs
    detail['diff'] = [segs[i]['mean'] - ssegs[i]['mean'] for i in range(4)]
    if lose:
        return False, f'有段输给 SHUF（第 {lose} 段）', detail

    return True, '四段同号且每段赢 SHUF', detail


def fmt(detail):
    """人读格式"""
    lines = []
    if 'segs' in detail and detail['segs']:
        lines.append('  段   笔数     真牌均值       t')
        for i, s in enumerate(detail['segs'], 1):
            lines.append(f'   {i}  {s["n"]:>6}  {s["mean"]*100:>+9.4f}%  {s["t"]:>+7.2f}')
    if 'diff' in detail:
        d = detail['diff']
        lines.append('  净信号(真−乱): ' + '  '.join(f'段{i+1} {x*100:+.4f}%'
                                                   for i, x in enumerate(d)))
    return '\n'.join(lines)
