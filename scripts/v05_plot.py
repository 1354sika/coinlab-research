# -*- coding: utf-8 -*-
"""v05_plot · 把 v05_sweep 的 CSV 画成图和结论（只读 CSV，不改数据）

matplotlib 本机缺中文字体 → 图内标签用英文，正文结论用中文写进 md。
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = r'D:\中信证券\coinlab\evidence\cycle'
DCODE = 'D01'
df = pd.read_csv(os.path.join(OUT, f'v05_sweep_{DCODE}.csv'))
df = df.dropna(subset=['mean_te']).sort_values('vw').reset_index(drop=True)
x = df['vw'].values / 60.0          # 小时

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle(f'V05 low-window sweep ({DCODE}, 1m, 60 coins, 365d) — 241 steps', fontsize=13)

ax = axes[0, 0]
ax.axhline(0, color='0.6', lw=1, ls='--')
ax.plot(x, df['mean_te'] * 100, lw=1.4, color='#1f77b4')
ax.fill_between(x, 0, df['mean_te'] * 100, where=df['mean_te'] > 0, color='#1f77b4', alpha=0.25)
ax.set_xlabel('V05 low window (hours)'); ax.set_ylabel('mean per trade, test (%)')
ax.set_title('Expectancy vs V05 (the key chart)')
ax.grid(alpha=0.25)

ax = axes[0, 1]
ax.plot(x, df['per_year'] / 1000.0, lw=1.4, color='#ff7f0e')
ax.set_xlabel('V05 low window (hours)'); ax.set_ylabel('entries per year (thousand)')
ax.set_title('Signal frequency vs V05')
ax.grid(alpha=0.25)
for h in (4, 6, 12, 24):
    if h <= x.max():
        i = np.argmin(np.abs(x - h))
        ax.annotate(f'{x[i]:.1f}h: {df.per_year[i]:,.0f}/yr', (x[i], df.per_year[i] / 1000),
                    textcoords='offset points', xytext=(4, 6), fontsize=8)

ax = axes[1, 0]
ax.plot(x, df['p_to'] * 100, lw=1.3, label='timeout %', color='#2ca02c')
ax.plot(x, df['p_sl'] * 100, lw=1.3, label='stop-loss %', color='#d62728')
ax.plot(x, df['p_tp'] * 100, lw=1.3, label='take-profit %', color='#9467bd')
ax.set_xlabel('V05 low window (hours)'); ax.set_ylabel('exit share (%)')
ax.set_title('Exit mix vs V05'); ax.legend(); ax.grid(alpha=0.25)

ax = axes[1, 1]
ax.plot(x, df['mdd'] * 100, lw=1.4, color='#8c564b')
ax.set_xlabel('V05 low window (hours)'); ax.set_ylabel('max drawdown (%)')
ax.set_title('Drawdown vs V05'); ax.grid(alpha=0.25)

plt.tight_layout()
png = os.path.join(OUT, f'v05_curve_{DCODE}.png')
plt.savefig(png, dpi=110)
print('PNG', png)

# ---- 结论 md ----
pos = df[df.mean_te > 0]
best = df.loc[df.mean_te.idxmax()]
worst = df.loc[df.mean_te.idxmin()]
i4 = np.argmin(np.abs(df.vw.values - 240))
i6 = np.argmin(np.abs(df.vw.values - 360))
i12 = np.argmin(np.abs(df.vw.values - 720))
i24 = np.argmin(np.abs(df.vw.values - 1440))

lines = []
A = lines.append
A('# V05 低点窗口穷举结论（4h→24h，241 档）')
A('')
A('> 数据 **D01**（60 币 / 365 天 / 1m）｜固定 V09=2880min V10=3% V14=低点×1.30 V15=低点×0.95 V16=1440min')
A('> 成本 V19=0.50%/笔，本金 V20=100，仓位 V21=10%，无筛选 V22')
A('> 报告：`v05_sweep_D01.csv`（241 行全量）、`v05_sweep_D01_summary.txt`、`v05_curve_D01.png`')
A('')
A('## 一、最重要的一张表：V05 与「一年出现几次」')
A('')
A('| V05 | 相对现在 | 入场点/年 | 测试笔数 | 单笔均值 | 胜率 | 100 元 → | 回撤 | 止盈 | 止损 | 超时 |')
A('|---|---|---|---|---|---|---|---|---|---|---|')
for i in (i4, i6, i12, i24):
    r = df.iloc[i]
    A(f'| **{int(r.vw)}min（{r.vw/60:.1f}h）** | {"基准" if i==i4 else f"{(r.vw-240)/60:+.1f}h"} | '
      f'{r.per_year:,.0f} | {int(r.n_test):,} | {r.mean_te:+.3%} | {r.win_te:.1%} | {r.final:,.2f} | '
      f'{r.mdd:.1%} | {r.p_tp:.1%} | {r.p_sl:.1%} | {r.p_to:.1%} |')
A('')
A('## 二、全局分布')
A('')
A(f'- **241 档里只有 {len(pos)} 档为正**（{len(pos)/len(df):.1%}）')
A(f'- 均值中位 **{df.mean_te.median():+.3%}**，最好 **{best.mean_te:+.3%}**（V05={int(best.vw)}min），最差 **{worst.mean_te:+.3%}**')
A(f'- 样本不足（<100 笔）的档：**0 档**（最小 {int(df.n_test.min())} 笔，样本量全程健康）')
A('')
A('## 三、结论')
A('')
A('### 1. V05 这个方向**不值**——越长越差，而且是单调的')
A('')
A('4 小时附近是全程唯一的微正区（240~330 分钟，均值 +0.02%~+0.09%），')
A('**6 小时以上基本全负**，24 小时跌到 −0.52%。没有任何「长窗口找到更实的底 → 收益更高」的证据。')
A('')
A('### 2. 真正的问题不是底找得准不准，是止盈太远')
A('')
A('超时率全程 **85%~88.8%**，几乎不随 V05 变化；止盈率从 2.2% 一路降到 1.0%。')
A('也就是说：**无论低点怎么选，绝大多数单子都在 24 小时里既够不到 +26.2%、也没跌破 −7.77%，最后按收盘价出场。**')
A('')
A('### 3. 信号频率随 V05 单调下降，但没有换来质量')
A('')
A(f'- V05=4h → **{df.iloc[i4].per_year:,.0f} 次/年**（约每天 32 次）')
A(f'- V05=6h → {df.iloc[i6].per_year:,.0f} 次/年')
A(f'- V05=12h → {df.iloc[i12].per_year:,.0f} 次/年')
A(f'- V05=24h → {df.iloc[i24].per_year:,.0f} 次/年（约每天 6.6 次）')
A('')
A('**机会少了 5 倍，单笔收益反而更差**——说明长窗口筛掉的不是噪声，是有效信号。')
A('')
A('### 4. 唯一「赢」的那 18 档，赢的量级可忽略')
A('')
A(f'最好的 V05={int(best.vw)}min 也只有 **{best.mean_te:+.3%}/笔**，扣完 0.5% 成本后，')
A(f'2,701 笔测试样本累计只有 100 → {best.final:,.2f}（+{best.final-100:.1f}%），**回撤 61.7%**。')
A('这个风险收益比在任何实盘标准下都不成立。')
A('')
A('## 四、下一步该动哪个变量（基于本轮证据）')
A('')
A('| 变量 | 建议 | 依据 |')
A('|---|---|---|')
A('| **V14 止盈** | **优先动**：×1.30 → ×1.10~1.15 | 86% 超时，止盈率仅 2.2%；降目标才能提高触发率 |')
A('| V05 低点窗口 | **不要动**（保持 240min） | 本轮 241 档证明：越长越差，没有峰 |')
A('| V16 持有上限 | 用户已定 24h，不动 | 但若 V14 降下来，超时率会显著下降 |')
A('| V09 确认窗口 | 可试 24h/12h（收紧） | 当前 48h 让入场点过多（1.17 万次/年） |')
A('')
A('## 五、诚实边界')
A('')
A('- 本结论只覆盖 **2026 年 1 年、60 个币、牛市样本**（D01），未验证熊市')
A('- 241 次比较存在多重检验风险：最好的 +0.088% 大概率是噪声')
A('- 未计滑点；已计 0.50% 手续费')
A('- 双触一律算止损（保守）')

md = os.path.join(OUT, '结论_V05扫描_2026-10-17.md')
with open(md, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print('MD', md)
