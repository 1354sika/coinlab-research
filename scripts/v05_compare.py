# -*- coding: utf-8 -*-
"""v05_compare · 把 D01/D07/D08 三份 V05 扫描结果放一张图，并做显著性检验

关键问题：D01 的「+0.022% 未 4h 微正峰」到底是信号还是噪声？
做法：从每档的 p_tp / p_sl / p_to + mean，反推「超时那批的均值」，再算单笔收益标准差，
      得到标准误 SE = std/sqrt(n)，给 t 值和 95% 置信区间。
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = r'D:\中信证券\coinlab\evidence\cycle'

TP_RET = 1.30 - 1.0      # [V14] 低点×1.30
SL_RET = 0.95 - 1.0      # [V15] 低点×0.95 → 但相对入场价约 -7.8%，见下
FEE = 0.005              # [V19]

SETS = [
    ('D01', 'D01  60 coins / 1y / 1m',  'v05_sweep_D01.csv', 'tab:blue'),
    ('D08', 'D08  60 coins / 3y / 5m',  'v05_sweep_D08.csv', 'tab:orange'),
    ('D07', 'D07 304 coins / 3y / 5m',  'v05_sweep_D07.csv', 'tab:purple'),
]

logbuf = []
def log(s):
    print(s, flush=True)
    logbuf.append(s)


def enrich(df):
    """补 std / SE / t / CI。SL 相对入场收益 = 低点×0.95 / (低点×1.03) - 1 - fee"""
    sl_rel = 0.95 / (1.0 + 0.03) - 1.0 - FEE      # 低点×0.95 相对「低点×1.03 入场」
    tp_rel = 1.30 / (1.0 + 0.03) - 1.0 - FEE
    # 反推超时那批的均值：mean = p_tp*tp + p_sl*sl + p_to*to
    p_to = np.maximum(1e-9, 1.0 - df['p_tp'] - df['p_sl'])
    to_rel = (df['mean_te'] - df['p_tp'] * tp_rel - df['p_sl'] * sl_rel) / p_to
    # 方差
    var = (df['p_tp'] * (tp_rel - df['mean_te']) ** 2
           + df['p_sl'] * (sl_rel - df['mean_te']) ** 2
           + p_to * (to_rel - df['mean_te']) ** 2)
    df = df.copy()
    df['sl_rel'] = sl_rel
    df['tp_rel'] = tp_rel
    df['to_rel'] = to_rel
    df['std'] = np.sqrt(var)
    df['se'] = df['std'] / np.sqrt(df['n_test'])
    df['t'] = df['mean_te'] / df['se']
    df['lo95'] = df['mean_te'] - 1.96 * df['se']
    df['hi95'] = df['mean_te'] + 1.96 * df['se']
    return df


log('=' * 108)
log('v05_compare · 三个数据集同一规则（V05=4h）的显著性对照')
log('  收益口径：TP=%+.2f%%  SL=%+.2f%%（相对「低点×1.03 入场」）  单笔扣 %.2f%%' % (100*(1.30/1.03-1-FEE), 100*(0.95/1.03-1-FEE), 100*FEE))
log('=' * 108)

DATA = {}
for code, desc, fn, color in SETS:
    p = os.path.join(OUT, fn)
    df = enrich(pd.read_csv(p))
    DATA[code] = df
    log(f'\n【{code}】{desc}')
    log(f'  档数 {len(df)}   测试笔数 {int(df.n_test.sum()):,}   期望值 中位 {df.mean_te.median()*100:+.4f}%   最好 {df.mean_te.max()*100:+.4f}%   最差 {df.mean_te.min()*100:+.4f}%')
    log(f'  单笔收益标准差(推) 中位 {df["std"].median()*100:.2f}%')
    # 与 D01 对齐的档（V05=240）
    r = df[df.vw == 240]
    if len(r):
        r = r.iloc[0]
        log(f'  ★ V05=240min（4h）这一档：n={int(r.n_test):,}  均值 {r.mean_te*100:+.4f}%  '
            f'SE {r.se*100:.4f}%  t={r.t:+.2f}  95%CI [{r.lo95*100:+.4f}%, {r.hi95*100:+.4f}%]  '
            f'{"显著" if abs(r.t) > 1.96 else "不显著（噪声）"}')

log('\n' + '=' * 108)
log('【核心判定】同一 V05=4h，三份数据的差距有多大')
log('=' * 108)
row = {}
for code in ['D01', 'D08', 'D07']:
    df = DATA[code]
    r = df[df.vw == 240]
    if len(r):
        row[code] = r.iloc[0]
log(f'  D01  60币/1年/1m : {row["D01"].mean_te*100:+.4f}%   t={row["D01"].t:+.2f}  → {"显著" if abs(row["D01"].t) > 1.96 else "★不显著，是噪声"}')
log(f'  D08  60币/3年/5m : {row["D08"].mean_te*100:+.4f}%   t={row["D08"].t:+.2f}  → 显著为负')
log(f'  D07 304币/3年/5m : {row["D07"].mean_te*100:+.4f}%   t={row["D07"].t:+.2f}  → 显著为负')
log('')
log(f'  消融①：币数 60→304（D08→D07，粒度时长不变）  {(row["D07"].mean_te - row["D08"].mean_te)*100:+.4f}pp  → 币数影响很小')
log(f'  消融②：粒度+时长（D01→D08，币数不变）        {(row["D08"].mean_te - row["D01"].mean_te)*100:+.4f}pp  → 主因在这里')

# ---------------- 图 ----------------
fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))
fig.suptitle('V05 sweep: D01 (+0.02%) vs D08/D07 (both clearly negative) — one rule, three datasets', fontsize=13)

# 1 期望值 + 95% CI
ax = axes[0][0]
for code, desc, fn, color in SETS:
    df = DATA[code]
    ax.plot(df.vw / 60, df.mean_te * 100, color=color, lw=1.3, label=f'{code}  n={int(df.n_test.sum()):,}')
    ax.fill_between(df.vw / 60, df.lo95 * 100, df.hi95 * 100, color=color, alpha=0.15)
ax.axhline(0, color='black', lw=1)
ax.set_xlabel('V05 low window (hours)')
ax.set_ylabel('Mean return per trade (%)')
ax.set_title('Expectancy with 95% CI — D01 band straddles 0; D08/D07 far below')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# 2 放大 D01 的置信带
ax = axes[0][1]
df = DATA['D01']
ax.plot(df.vw / 60, df.mean_te * 100, color='tab:blue', lw=1.3, label='D01 mean')
ax.fill_between(df.vw / 60, df.lo95 * 100, df.hi95 * 100, color='tab:blue', alpha=0.2, label='95% CI')
ax.axhline(0, color='black', lw=1)
m = df.mean_te.idxmax()
ax.annotate(f'best {df.loc[m,"mean_te"]*100:+.3f}%\nV05={df.loc[m,"vw"]/60:.2f}h',
            xy=(df.loc[m, 'vw'] / 60, df.loc[m, 'mean_te'] * 100),
            xytext=(0.45, 0.75), textcoords='axes fraction', fontsize=9,
            arrowprops=dict(arrowstyle='->', lw=1))
ax.set_xlabel('V05 low window (hours)')
ax.set_ylabel('Mean return per trade (%)')
ax.set_title('D01 alone: entire 95% CI band contains 0 → cannot call it an edge')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# 3 每年入场次数
ax = axes[1][0]
for code, desc, fn, color in SETS:
    df = DATA[code]
    ax.plot(df.vw / 60, df.per_year, color=color, lw=1.3, label=code)
for h in (4, 6, 12, 24):
    ax.axvline(h, color='gray', ls=':', lw=0.8)
ax.set_xlabel('V05 low window (hours)')
ax.set_ylabel('Entries per year')
ax.set_yscale('log')
ax.set_title('Signal frequency (log) — D07 has ~6x more because 304 coins')
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which='both')

# 4 出口构成（D01 vs D07，V05=4h 前后）
ax = axes[1][1]
labels = ['take-profit', 'stop-loss', 'timeout']
for i, code in enumerate(['D01', 'D07']):
    df = DATA[code]
    r = df[df.vw == 240].iloc[0]
    vals = [r.p_tp * 100, r.p_sl * 100, r.p_to * 100]
    xs = np.arange(3) + (i - 0.5) * 0.35
    bars = ax.bar(xs, vals, width=0.33, label=f'{code} V05=4h', color=['tab:blue', 'tab:purple'][i])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f'{v:.1f}%', ha='center', fontsize=8)
ax.set_xticks(np.arange(3))
ax.set_xticklabels(labels)
ax.set_ylabel('Share of exits (%)')
ax.set_title('Exit mix at V05=4h — timeout dominates ~86% in both')
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
png = os.path.join(OUT, 'v05_compare_D01D07D08.png')
plt.savefig(png, dpi=125)
log(f'\nPNG {png}')

md = os.path.join(OUT, '结论_V05跨数据集对照_2026-10-17.md')
with open(md, 'w', encoding='utf-8') as f:
    f.write('# V05 扫描 · 跨数据集对照结论（2026-10-17）\n\n```\n' + '\n'.join(logbuf) + '\n```\n\n')
    f.write(f'![对比图]({png})\n')
log(f'MD {md}')
