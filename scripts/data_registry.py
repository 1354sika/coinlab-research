# -*- coding: utf-8 -*-
"""data_registry · D 系列数据编号的唯一真源（见 数据资产编号表_v1.0.md）

规则：脚本里**禁止再写死数据文件名**，一律用 D 编号经此模块解析。
这样「哪份数据」和「哪些变量」两套编号互不污染，报告里写 D07 就永远指同一份文件。
"""
import os

DATA = r'D:\中信证券\coinlab\data'

# D 编号 -> (文件名, 每根多少分钟, 说明, 能否跑一号算法)
DMAP = {
    'D01': ('klines_1m_1y.csv',            1, '60 币 / 365 天 / 1m',      True),
    'D02': ('klines_1m_502.csv',           1, '502 币 / 60 天 / 1m',      True),
    'D03': ('klines_1m_wide.csv',          1, '100 币 / 142 天 / 1m',     True),
    'D04': ('klines_1m_sync.csv',          1, '100 币 / 60 天 / 1m',      True),
    'D05': ('klines_1m_long.csv',          1, '12 币 / 39 天 / 1m',       True),
    'D06': ('klines_1m_top20_7d.csv',      1, '20 币 / 7 天 / 1m',        True),
    'D07': ('klines_5m_502.csv',           5, '304 币 / 1095 天 / 5m',    True),
    'D08': ('klines_5m_3y.csv',            5, '60 币 / 1095 天 / 5m',     True),
    'D09': ('klines_1d_full.csv',       1440, '502 币 / 3346 天 / 1d',    False),
    'D10': ('klines_1d_top502.csv',     1440, '502 币 / 1006 天 / 1d',    False),
    'D11': ('klines_1d_top50.csv',      1440, '50 币 / 999 天 / 1d',      False),
    'D12': ('klines_1h_top20_365d.csv',   60, '20 币 / 365 天 / 1h',      False),
}


def resolve(dcode, need_algo=True):
    """把 D 编号解析成 (dcode, 绝对路径, 每根分钟数, 说明)。编号非法直接退出，绝不猜。"""
    if dcode not in DMAP:
        raise SystemExit(f'[D 编号错] 未知 {dcode!r}；可用：{", ".join(sorted(DMAP))}')
    fn, gran, desc, algo_ok = DMAP[dcode]
    if need_algo and not algo_ok:
        raise SystemExit(
            f'[D 编号错] {dcode} = {desc}：粒度太粗（V05=240min 不足 1 根），不能跑一号算法')
    path = os.path.join(DATA, fn)
    if not os.path.exists(path):
        raise SystemExit(f'[D 编号错] {dcode} 文件不存在：{path}')
    return dcode, path, gran, desc


def list_codes():
    return sorted(DMAP.keys())
