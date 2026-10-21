# -*- coding: utf-8 -*-
"""paper_trade.py — 单账号·满仓复利·纸面前向验证器

用途：不用真钱，从"现在"开始记录这套规则的真实表现，验证它是否真的能打。
      回测已做尽（3 年 2,906 笔、9/9 季正、59/60 币正），
      但回测无法证明「2% 止盈单能否成交」——纸面前向记录是唯一出路。

规则（与 best_curve / tp_verify 完全一致）：
  入场：ret24 <= -2%  且  volr >= 3.0（volr = 当前 5m 量 / 前 24 根均量）
  出场：首次收益 >= +2% 即卖；否则 θ=1% 跟踪止盈（从峰值回落 1%）；最长持 96 根（8h）
  成本：0.04% 单边（买卖各一次）
  仓位：单账号、一次一笔、满仓、复利（净值滚动）

运行：
  python -X utf8 paper_trade.py --once     # 检查一次（适合定时任务/cron）
  python -X utf8 paper_trade.py --loop 60  # 每 60 秒循环检查

数据来源：coinlab/data/live/<UTC日期>/klines_1s.csv（live_recorder 正在写）
状态文件：coinlab/evidence/paper_state.json
流水账：coinlab/evidence/paper_ledger.csv

诚实边界：
  · 纸面成交按「信号出现后的下一根 5m 开盘价」买入、按「触价 2% 的 close」卖出；
    实盘存在排队与滑点，本脚本不做成交率假设，只如实记录。
  · 单账号一次一笔，若信号出现时已有持仓则跳过（会漏掉并发机会，这就是真实约束）。
"""
import os
import sys
import json
import time
import glob
import numpy as np
import pandas as pd

ROOT = r'D:\中信证券\coinlab'
LIVE = os.path.join(ROOT, 'data', 'live')
EVI = os.path.join(ROOT, 'evidence')
STATE = os.path.join(EVI, 'paper_state.json')
LEDGER = os.path.join(EVI, 'paper_ledger.csv')

DIP, VOLR, TP, THETA, COST = -0.02, 3.0, 0.02, 0.01, 0.0004
MAX_HOLD = 96          # 5m 根数 = 8 小时
INIT_CASH = 1000.0     # 初始净值（纸面）


def log(*a):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), *a, flush=True)


def load_state():
    if os.path.exists(STATE):
        with open(STATE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {'cash': INIT_CASH, 'equity': INIT_CASH, 'position': None,
            'peak_equity': INIT_CASH, 'max_dd': 0.0, 'n_trades': 0, 'n_win': 0,
            'started': time.strftime('%Y-%m-%d %H:%M:%S')}


def save_state(s):
    os.makedirs(EVI, exist_ok=True)
    with open(STATE, 'w', encoding='utf-8') as f:
        json.dump(s, f, ensure_ascii=False, indent=2)


def append_ledger(row):
    df = pd.DataFrame([row])
    hdr = not os.path.exists(LEDGER)
    df.to_csv(LEDGER, mode='a', header=hdr, index=False, encoding='utf-8')


def load_5m():
    """读最近两个日期目录的 1s 数据，聚合成 5m，返回 {sym: DataFrame[5m]}"""
    days = sorted(glob.glob(os.path.join(LIVE, '2*')), reverse=True)[:2]
    frames = []
    for d in days:
        p = os.path.join(d, 'klines_1s.csv')
        if not os.path.exists(p):
            continue
        try:
            df = pd.read_csv(p, skiprows=1, header=None)
        except Exception:
            continue
        n = df.shape[1]
        df.columns = ['open_time', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'close_time'][:n]
        for c in ['open_time', 'open', 'high', 'low', 'close', 'volume']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.dropna(subset=['open_time', 'close', 'volume'])
        df['open_time'] = df['open_time'].astype('int64')
        frames.append(df)
    if not frames:
        return {}
    df = pd.concat(frames, ignore_index=True)
    df['bucket'] = df['open_time'] // 300_000
    agg = (df.groupby(['symbol', 'bucket'])
             .agg(open=('open', 'first'), high=('high', 'max'), low=('low', 'min'),
                  close=('close', 'last'), volume=('volume', 'sum'))
             .reset_index().sort_values('bucket'))
    return {sym: g.reset_index(drop=True) for sym, g in agg.groupby('symbol')}


def data_lag_min(bars):
    """最新一根 5m 距今多少分钟（bucket = epoch 秒 // 300）"""
    if not bars:
        return None
    mx = max(int(g.bucket.iloc[-1]) for g in bars.values())
    return (time.time() // 300 - mx) * 5.0


def compute_signals(bars):
    """返回 [(sym, bucket, close, ret24, volr)]，只用已完成的 5m 根"""
    out = []
    for sym, g in bars.items():
        n = len(g)
        if n < 26:
            continue
        c = g.close.to_numpy(np.float64)
        v = g.volume.to_numpy(np.float64)
        vma = pd.Series(v).rolling(24, min_periods=24).mean().to_numpy()
        volr = np.where(vma > 0, v / vma, 1.0)
        ret24 = c[-1] / c[-25] - 1.0
        if ret24 <= DIP and volr[-1] >= VOLR:
            out.append((sym, int(g.bucket.iloc[-1]), float(c[-1]), float(ret24), float(volr[-1])))
    return out


def check_exit(state, bars):
    """若持仓，检查是否该卖（TP 或 θ 回撤）"""
    pos = state['position']
    if not pos:
        return state
    sym = pos['sym']
    if sym not in bars:
        return state
    g = bars[sym]
    g = g[g.bucket >= pos['bucket']]
    if len(g) == 0:
        return state
    entry = pos['entry']
    peak = max(pos['peak_px'], float(g.high.max()))
    close_now = float(g.close.iloc[-1])
    bars_held = len(g)
    reason = None
    exit_px = None

    # 先看有没有触及 +2%
    hit = g[g.high >= entry * (1 + TP)]
    if len(hit):
        reason = 'TP2%'
        exit_px = float(hit.close.iloc[0])
    elif peak > 0 and close_now <= peak * (1 - THETA) and bars_held >= 2:
        reason = 'theta1%'
        exit_px = close_now
    elif bars_held >= MAX_HOLD:
        reason = 'timeout8h'
        exit_px = close_now

    if reason:
        gross = exit_px / entry - 1.0
        net = gross - 2 * COST
        state['cash'] = state['cash'] * (1 + net)
        state['equity'] = state['cash']
        state['n_trades'] += 1
        if net > 0:
            state['n_win'] += 1
        state['peak_equity'] = max(state['peak_equity'], state['equity'])
        state['max_dd'] = min(state['max_dd'], state['equity'] / state['peak_equity'] - 1.0)
        append_ledger({'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'sym': sym,
                       'entry': round(entry, 8), 'exit': round(exit_px, 8),
                       'gross': round(gross, 6), 'net': round(net, 6),
                       'reason': reason, 'bars_held': bars_held,
                       'equity': round(state['equity'], 4)})
        log(f'平仓 {sym} {reason} net={net:+.4%} equity={state["equity"]:.4f}')
        state['position'] = None
    else:
        state['position']['peak_px'] = peak
    return state


def step():
    state = load_state()
    bars = load_5m()
    if not bars:
        log('无 live 数据（live_recorder 未写出？）')
        return state
    state = check_exit(state, bars)

    n_sig = 0
    if state['position'] is None:
        sigs = compute_signals(bars)
        n_sig = len(sigs)
        if sigs:
            # 选量比最大的一个（单账号一次一笔）
            sigs.sort(key=lambda x: -x[4])
            sym, bucket, px, ret24, volr = sigs[0]
            state['position'] = {'sym': sym, 'entry': px, 'bucket': bucket,
                                 'peak_px': px, 'ret24': ret24, 'volr': volr,
                                 'opened': time.strftime('%Y-%m-%d %H:%M:%S')}
            log(f'开仓 {sym} @ {px:.8f} ret24={ret24:.2%} volr={volr:.2f} '
                f'(同期信号 {len(sigs)} 个，选量比最大)')
    save_state(state)
    lag = data_lag_min(bars)
    alert = '' if (lag is not None and lag <= 30) else '  <<ALERT 数据源停滞>>'
    log(f'心跳 净值={state["equity"]:.4f} 笔数={state["n_trades"]} '
        f'持仓={state["position"]["sym"] if state["position"] else "无"} '
        f'数据滞后={lag:.0f}分 信号={n_sig}{alert}')
    return state


def report():
    s = load_state()
    n = s['n_trades']
    log('=' * 60)
    log(f'纸面验证进度：{n} 笔  净值 {s["equity"]:.4f}（初始 {INIT_CASH}）')
    if n:
        log(f'  胜率 {s["n_win"]/n:.2%}  最大回撤 {s["max_dd"]:.2%}  '
            f'平均每笔 {(s["equity"]/INIT_CASH)**(1/n)-1:.4%}')
    log(f'  持仓：{s["position"] if s["position"] else "无"}')
    log(f'  起始：{s["started"]}')
    log('=' * 60)


if __name__ == '__main__':
    if '--once' in sys.argv:
        step(); report()
    elif '--loop' in sys.argv:
        every = int(sys.argv[sys.argv.index('--loop') + 1])
        log(f'循环启动，每 {every} 秒检查一次')
        while True:
            try:
                step()
            except Exception as e:
                log(f'异常: {e}')
            time.sleep(every)
    elif '--report' in sys.argv:
        report()
    else:
        step(); report()
