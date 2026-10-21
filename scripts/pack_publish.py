# -*- coding: utf-8 -*-
"""pack_publish.py — 把研究成果打包成可发布的 git 仓库（排除大文件）"""
import os
import shutil

SRC = r'D:\中信证券\coinlab'
DST = r'D:\中信证券\coinlab-publish'
KEEP = {'.txt', '.md', '.png', '.json', '.svg'}


def main():
    if os.path.exists(DST):
        shutil.rmtree(DST)
    os.makedirs(os.path.join(DST, 'scripts'))
    os.makedirs(os.path.join(DST, 'evidence'))

    ns = ss = 0
    for f in os.listdir(os.path.join(SRC, 'scripts')):
        if f.endswith('.py'):
            s = os.path.join(SRC, 'scripts', f)
            shutil.copy2(s, os.path.join(DST, 'scripts', f))
            ns += 1
            ss += os.path.getsize(s)

    ne = se = 0
    base = os.path.join(SRC, 'evidence')
    for root, _dirs, files in os.walk(base):
        rel = os.path.relpath(root, base)
        for f in files:
            if os.path.splitext(f)[1].lower() in KEEP:
                s = os.path.join(root, f)
                d = os.path.join(DST, 'evidence', f) if rel == '.' else os.path.join(DST, 'evidence', rel, f)
                os.makedirs(os.path.dirname(d), exist_ok=True)
                shutil.copy2(s, d)
                ne += 1
                se += os.path.getsize(s)

    print('scripts %d files %.2f MB' % (ns, ss / 1048576))
    print('evidence %d files %.2f MB' % (ne, se / 1048576))
    print('DST =', DST, 'exists =', os.path.exists(DST))


if __name__ == '__main__':
    main()
