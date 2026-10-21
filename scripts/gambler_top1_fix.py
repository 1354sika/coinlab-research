# -*- coding: utf-8 -*-
"""gambler_top1_fix · 修正「去 top1%」的实现错误

上一版（verify2）判「✗靠极端笔」是**错的**，原因是实现 bug：
    q = g.ret.quantile(0.99); g2 = g[g.ret < q]
收益是**离散的**（大量笔恰好等于止盈值 +29.91% 或止损值 -15.09%），
C 组止盈笔占 16.4% > 1%，所以 quantile(0.99) 就等于 +29.91%，
`ret < q` 把**全部 260 笔止盈**都删掉了，而不是只删 1%（16 笔）。

验算 C 组：
    删前 (7127 百分点) - 260×29.91 (=7777) / 1327 笔 = -0.49%  ← 与报告一致
    证实是"删了 16.4% 而不是 1%"

正确做法：按**排名**精确删最高 n//100 笔。

本脚本在已有 CSV 上重算（不重跑扫描）。
"""
import os
import numpy as np
import pandas as pd

OUT = r'D:\中信证券\coinlab\evidence\cycle'
SRC = os.path.join(OUT, 'bidir_verify2_rows.csv')
logbuf = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); logbuf.append(s)

log('=' * 116)
log('gambler_top1_fix · 修正「去 top1%」实现错误')
log('=' * 116)

df = pd.read_csv(SRC)
log(f'读入 {len(df):,} 笔')

log('')
log('=' * 116)
log('【一】正确版：按排名精确删最高 1%（而非按阈值）')
log('=' * 116)
log(f'{"条件":<22} {"笔数":>8} {"原期望%":>9} {"删1%后%":>10} {"差":>8}  判定')
log('-' * 116)
for name in df.cond.unique():
    g = df[df.cond == name].sort_values('ret', ascending=False)
    n = len(g)
    k = max(1, n // 100)
    g2 = g.iloc[k:]
    e0 = g.ret.mean(); e1 = g2.ret.mean()
    ok = '★通过（不靠极端笔）' if e1 > 0 else '✗仍靠极端笔'
    log(f'{name:<22} {n:>8,} {e0:>8.2f}% {e1:>9.2f}% {e0-e1:>7.2f}%  {ok}')

log('')
log('=' * 116)
log('【二】参考：删最高 1% / 2% / 5% 各档（看衰减速度）')
log('=' * 116)
log(f'{"条件":<22} {"原":>8} {"-1%":>8} {"-2%":>8} {"-5%":>8} {"-10%":>8}')
log('-' * 116)
for name in df.cond.unique():
    g = df[df.cond == name].sort_values('ret', ascending=False)
    n = len(g)
    row = [g.ret.mean()]
    for pct in [1, 2, 5, 10]:
        k = max(1, n * pct // 100)
        row.append(g.iloc[k:].ret.mean())
    log(f'{name:<22} ' + ' '.join(f'{v:>7.2f}%' for v in row))

log('')
log('=' * 116)
log('【三】收益构成（解释为什么删止盈会崩：收益是离散三档）')
log('=' * 116)
for name in df.cond.unique():
    g = df[df.cond == name]
    tp = (g.ret > 29.0).mean()
    sl = (g.ret < -14.5).mean()
    mid = 1 - tp - sl
    log(f'{name}')
    log(f'   止盈(+29.91%) 占 {100*tp:5.1f}%   止损(-15.09%) 占 {100*sl:5.1f}%   超时平仓 占 {100*mid:5.1f}%')
    gm = g[(g.ret <= 29.0) & (g.ret >= -14.5)]
    if len(gm):
        log(f'   超时平仓那批：笔 {len(gm):,}  均值 {gm.ret.mean():+.2f}%  中位 {gm.ret.median():+.2f}%')
    log('')

log('=' * 116)
log('【四】结论')
log('=' * 116)
log('  上一轮「✗靠极端笔」的判定作废——那是删了 16.4% 的止盈笔，不是 1%。')
log('  真实情况：收益由三档构成（止盈/止损/超时），止盈笔占比 3.8%~18.8%，')
log('  是**成百上千笔**在贡献，不是十几笔极端值。')

os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, 'top1_fix_summary.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(logbuf))
log('')
log(f'报告已写 {os.path.join(OUT, "top1_fix_summary.txt")}')
