# -*- coding: utf-8 -*-
"""把 evidence 下所有 txt/md 压成一份 digest：标题 + 关键词行 + 尾部"""
import os, io, re

ROOT = r'D:\中信证券\coinlab\evidence'
OUT = os.path.join(ROOT, '_digest_2026-10-17.md')
KW = re.compile(r'(结论|判读|★|✗|✅|期望|撤回|不成立|限制|注意|边界|建议|风险|失败|不显著|通过)')

rows = []
for dirpath, dirs, files in os.walk(ROOT):
    dirs[:] = [d for d in dirs if d not in ('__pycache__',)]
    for fn in sorted(files):
        if not fn.lower().endswith(('.txt', '.md')):
            continue
        p = os.path.join(dirpath, fn)
        rel = os.path.relpath(p, ROOT)
        try:
            lines = io.open(p, encoding='utf-8', errors='replace').read().splitlines()
        except Exception:
            continue
        if not lines:
            continue
        head = [l.strip() for l in lines[:6] if l.strip()]
        kw = []
        for l in lines:
            s = l.strip()
            if s and KW.search(s) and 8 < len(s) < 260:
                kw.append(s)
                if len(kw) >= 10:
                    break
        tail = [l.strip() for l in lines[-10:] if l.strip()]
        rows.append((rel.replace('\\', '/'), len(lines), head, kw, tail))

rows.sort()
with io.open(OUT, 'w', encoding='utf-8') as f:
    f.write(f'# evidence digest · {len(rows)} 份 txt/md\n')
    for rel, n, head, kw, tail in rows:
        f.write(f'\n## {rel} ({n}行)\n')
        if head:
            f.write('H| ' + ' | '.join(head[:3])[:400] + '\n')
        if kw:
            f.write('K| ' + ' || '.join(kw)[:900] + '\n')
        if tail:
            f.write('T| ' + ' | '.join(tail[-3:])[:400] + '\n')
print(f'{len(rows)} files -> {OUT}')
