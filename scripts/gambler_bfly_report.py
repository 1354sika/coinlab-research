# -*- coding: utf-8 -*-
"""gambler_bfly_report.py — 蝴蝶效应全量搜索的汇总与出图

读 butterfly/ 下所有 CSV（static_50_all / static_200_all / chain2_all /
chain3_sampled / chain4_sampled / cross_mkt_all / cross_coin_sampled），
产出一张总览图 + 一份 Markdown 汇总（含"哪部分真穷举、哪部分采样、覆盖率"）。

产物（落 D 盘）：
  D:\中信证券\coinlab\evidence\butterfly\bfly_overview.png
  D:\中信证券\coinlab\evidence\butterfly\蝴蝶效应_全量搜索汇总.md
"""
import os, glob, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EV = r'D:\中信证券\coinlab\evidence'
OUTDIR = os.path.join(EV, 'butterfly')
os.makedirs(OUTDIR, exist_ok=True)

FILES = {
    'static_50_all.csv':       ('静态·核心池50条件（2/3/4元）', '真穷举'),
    'static_200_all.csv':      ('静态·扩展池196条件（2/3元）',  '真穷举'),
    'chain2_all.csv':          ('链2段 A→τ→B（同币）',         '真穷举'),
    'chain3_sampled.csv':      ('链3段 A→τ→B→τ′→C（同币）',    '采样 36.6%'),
    'chain4_sampled.csv':      ('链4段（同币）',                '抽查 0.03%'),
    'cross_mkt_all.csv':       ('跨币·市场→τ→单币',            '真穷举'),
    'cross_coin_sampled.csv':  ('跨币·币→τ→币',                '采样 0.3%'),
}

rows = []
data = {}
for fn, (label, cover) in FILES.items():
    p = os.path.join(OUTDIR, fn)
    if not os.path.exists(p):
        rows.append((label, cover, None, None, None, None, None, '未产出'))
        continue
    d = pd.read_csv(p)
    if not len(d):
        rows.append((label, cover, 0, None, None, None, None, '空'))
        continue
    data[label] = d
    ok = d[np.isfinite(d['lift_test'])] if 'lift_test' in d else d.iloc[0:0]
    rows.append((label, cover, len(d),
                 float(d['lift_train'].max()) if 'lift_train' in d else np.nan,
                 float(ok['lift_test'].median()) if len(ok) else np.nan,
                 float(100 * (ok['lift_test'] > 5).mean()) if len(ok) else np.nan,
                 float(100 * (ok['lift_test'] < 1).mean()) if len(ok) else np.nan,
                 'OK'))

TAB = pd.DataFrame(rows, columns=['阶段', '覆盖方式', '组合数', '训练段最高lift',
                                  '测试段lift中位', '测试>5x占比%', '测试<1x(崩)占比%', '状态'])

# ---------- 出图 ----------
fig, axes = plt.subplots(2, 2, figsize=(16, 11))
plt.rcParams['axes.unicode_minus'] = False

# (1) 各阶段组合数（对数）
ax = axes[0][0]
t = TAB[TAB['组合数'].notna() & (TAB['组合数'] > 0)]
if len(t):
    ax.barh(range(len(t)), t['组合数'].values, color='#4a6fa5')
    ax.set_yticks(range(len(t))); ax.set_yticklabels([s[:22] for s in t['阶段']], fontsize=9)
    ax.set_xscale('log'); ax.set_xlabel('有效组合数（对数轴）')
    ax.set_title('各阶段搜到的有效组合数')
    for i, v in enumerate(t['组合数'].values):
        ax.text(v, i, f' {int(v):,}', va='center', fontsize=8)

# (2) train vs test lift 散点（所有阶段叠加）
ax = axes[0][1]
markers = ['o', 's', '^', 'v', 'D', 'P', '*']
for i, (label, d) in enumerate(data.items()):
    if 'lift_test' not in d: continue
    o = d[np.isfinite(d['lift_test'])]
    if not len(o): continue
    s = np.clip(o['n_train'].values / 500, 3, 60)
    ax.scatter(o['lift_train'], o['lift_test'], s=s, alpha=0.35,
               marker=markers[i % len(markers)], label=label[:20])
ax.axhline(1, color='#888', lw=1, ls='--')
ax.axvline(1, color='#888', lw=1, ls='--')
ax.set_xlabel('训练段 lift'); ax.set_ylabel('测试段 lift')
ax.set_title('训练段 vs 测试段 lift（对角线=无衰减）')
ax.legend(fontsize=7, loc='upper left')

# (3) 测试段 lift 分布（箱线）
ax = axes[1][0]
box, labs = [], []
for label, d in data.items():
    if 'lift_test' not in d: continue
    o = d[np.isfinite(d['lift_test'])]['lift_test'].values
    if len(o) < 5: continue
    box.append(np.clip(o, 0, 30)); labs.append(label[:16])
if box:
    bp = ax.boxplot(box, tick_labels=labs, vert=False, showfliers=False)
    ax.set_xlabel('测试段 lift'); ax.set_title('各阶段测试段 lift 分布（箱线，截到30）')
    ax.tick_params(axis='y', labelsize=8)

# (4) 汇总表
ax = axes[1][1]; ax.axis('off')
show = TAB.copy()
for c in ['训练段最高lift', '测试段lift中位', '测试>5x占比%', '测试<1x(崩)占比%']:
    show[c] = show[c].apply(lambda v: '—' if pd.isna(v) else f'{v:.1f}')
show['组合数'] = show['组合数'].apply(lambda v: '—' if pd.isna(v) else f'{int(v):,}')
tab = ax.table(cellText=show.values, colLabels=show.columns, loc='center', cellLoc='left')
tab.auto_set_font_size(False); tab.set_fontsize(7.5); tab.scale(1, 1.5)
ax.set_title('蝴蝶效应全量搜索 · 覆盖方式总表', fontsize=11)

plt.tight_layout()
png = os.path.join(OUTDIR, 'bfly_overview.png')
plt.savefig(png, dpi=110); plt.close()

# ---------- Markdown 汇总 ----------
L = []
L.append('# gambler · 蝴蝶效应全量搜索汇总')
L.append('')
L.append(f'**生成时间** {time.strftime("%Y-%m-%d %H:%M:%S")} ｜ **目录** `{OUTDIR}`')
L.append('')
L.append('## 覆盖声明（诚实边界，必须逐条读）')
L.append('')
L.append('**字面意义上的"所有可能的蝴蝶效应"不可能穷尽**——特征可以无限构造、'
         '时间间隔是连续量。本文档所做的是**在明确定义的组合空间内做到 100% 覆盖**，'
         '并用受控采样覆盖那些规模过大的空间。**每阶段的覆盖方式逐一标注如下**：')
L.append('')
L.append(TAB.to_markdown(index=False))
L.append('')
L.append('**解读**：')
L.append('- 「真穷举」= 该空间内**每个组合都算过**，不存在遗漏')
L.append('- 「采样 X%」= 该空间**只随机抽了 X%**，结论**不得**推广为"不存在其他组合"')
L.append('')
L.append('## 各阶段 Top 组合')
L.append('')
for label, d in data.items():
    if 'lift_test' not in d: continue
    L.append(f'### {label}')
    L.append('')
    L.append('| 组合 | 训练真局率 | 训练n | 测试真局率 | 测试n | 测试lift |')
    L.append('|---|---|---|---|---|---|')
    for _, r in d.head(12).iterrows():
        L.append(f'| `{r["combo"] if "combo" in d else r["chain"]}` '
                 f'| {100*r["rate_train"]:.2f}% | {int(r["n_train"]):,} '
                 f'| {100*r["rate_test"]:.2f}% | {int(r["n_test"]):,} '
                 f'| {r["lift_test"]:.1f}x |')
    L.append('')

L.append('## 判读')
L.append('')
L.append('- **测试段 lift 是唯一有效的指标**（训练段 lift 会被多重比较严重膨胀）')
L.append('- 「测试<1x(崩)占比」= 训练段排前面的组合里，有多少在测试段比基准还差 ⇒ 过拟合比例')
L.append('- 所有条件在 t 时刻（或 t−τ 时刻）**全部可观测**，无前视')

md = os.path.join(OUTDIR, '蝴蝶效应_全量搜索汇总.md')
with open(md, 'w', encoding='utf-8') as f:
    f.write('\n'.join(L))

print(f'saved {png}')
print(f'saved {md}')
print(TAB.to_string(index=False))
