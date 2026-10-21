# -*- coding: utf-8 -*-
"""health.py — 运行态一体体检（只读）

回答一个问题：现在这套自动化还在正常跑吗？
  1. 数据通道：data/live 最新 1s/盘口文件的写入新鲜度 + 最新 5m 根滞后
  2. 纸面验证器：paper_state.json 的净值/笔数/持仓 + 距 1.5x 进度
  3. 结论：OK / ALERT（哪条断了）

用法：python -X utf8 health.py
"""
import os
import json
import glob
import time

ROOT = r'D:\中信证券\coinlab'
LIVE = os.path.join(ROOT, 'data', 'live')
EVI = os.path.join(ROOT, 'evidence')
STATE = os.path.join(EVI, 'paper_state.json')
LEDGER = os.path.join(EVI, 'paper_ledger.csv')

LAG_ALERT_MIN = 30      # 最新 5m 根超过 30 分钟 = 数据通道可疑
FILE_ALERT_MIN = 40     # 文件 40 分钟没更新 = live_recorder 可能死了


def fmt_ts(ms):
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ms / 1000))


def check_files():
    days = sorted(glob.glob(os.path.join(LIVE, '2*')), reverse=True)[:1]
    if not days:
        return None, 'ALERT 无日期目录'
    d = days[0]
    lines = []
    worst = 0.0
    for name in ('klines_1s.csv', 'aggTrades.csv', 'depth20.csv'):
        p = os.path.join(d, name)
        if not os.path.exists(p):
            lines.append(f'{name}=缺失')
            worst = 999
            continue
        age_min = (time.time() - os.path.getmtime(p)) / 60.0
        worst = max(worst, age_min)
        lines.append(f'{name}={age_min:.0f}分前/{os.path.getsize(p)/1e6:.1f}MB')
    tag = 'OK' if worst <= FILE_ALERT_MIN else 'ALERT'
    return worst, f'[{tag}] {os.path.basename(d)} ' + '  '.join(lines)


def check_lag():
    """最新 5m 根距今分钟数（读 klines_1s 末行）"""
    days = sorted(glob.glob(os.path.join(LIVE, '2*')), reverse=True)[:1]
    if not days:
        return None
    p = os.path.join(days[0], 'klines_1s.csv')
    if not os.path.exists(p):
        return None
    # 从文件尾部读最后 8KB，取最后一条完整行
    with open(p, 'rb') as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - 8192))
        tail = f.read().decode('utf-8', 'ignore').strip().split('\n')
    latest = None
    for line in reversed(tail):
        parts = line.split(',')
        if len(parts) >= 2 and parts[0].isdigit():
            latest = int(parts[0])
            break
    if latest is None:
        return None
    # 毫秒 time？1s 数据 open_time 有时是秒
    ms = latest if latest > 1e12 else latest * 1000
    return (time.time() * 1000 - ms) / 60000.0


def check_paper():
    if not os.path.exists(STATE):
        return None, 'ALERT 无 paper_state.json'
    with open(STATE, 'r', encoding='utf-8') as f:
        s = json.load(f)
    n = s.get('n_trades', 0)
    eq = s.get('equity', 0)
    pos = s.get('position')
    win = s.get('n_win', 0)
    need = 0
    if 0 < eq < 1500:
        need = 200  # 未知，占位
    wr = (win / n) if n else None
    tag = 'OK'
    if n and wr is not None and wr < 0.55:
        tag = 'WARN 胜率偏低'
    return s, (f'[{tag}] 笔数={n} 净值={eq:.2f} 胜率={"-" if wr is None else f"{wr:.1%}"} '
               f'回撤={s.get("max_dd", 0):.2%} 持仓={pos["sym"] if pos else "无"} '
               f'距1.5x还差={1500/eq:.2f}x' if eq > 0 else '')


def main():
    try:
        import sys
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    print('=' * 72)
    print('运行态体检', time.strftime('%Y-%m-%d %H:%M:%S'))
    print('=' * 72)
    worst, ftxt = check_files()
    print('数据文件 :', ftxt)
    lag = check_lag()
    if lag is None:
        print('5m 滞后  : ALERT 读不到时间戳')
    else:
        tag = 'OK' if lag <= LAG_ALERT_MIN else 'ALERT 数据源停滞'
        print(f'5m 滞后  : {lag:.0f} 分 [{tag}]')
    s, ptxt = check_paper()
    print('纸面账户 :', ptxt)
    if os.path.exists(LEDGER):
        with open(LEDGER, 'r', encoding='utf-8', errors='ignore') as f:
            cnt = sum(1 for _ in f) - 1
        print(f'流水账   : {cnt} 行')
    alive = (worst is not None and worst <= FILE_ALERT_MIN) and (lag is not None and lag <= LAG_ALERT_MIN)
    print('-' * 72)
    print('结论     :', 'OK 全链正常，机器在盯' if alive else 'ALERT 有环节停滞，见上')
    print('=' * 72)


if __name__ == '__main__':
    import sys
    if '--watch' in sys.argv:
        every = int(sys.argv[sys.argv.index('--watch') + 1]) if len(sys.argv) > sys.argv.index('--watch') + 1 else 600
        while True:
            main()
            time.sleep(every)
    else:
        main()
