# -*- coding: utf-8 -*-
"""拉取守护：每 2 分钟检查 fetch_1m_all.py 是否还活着；挂了就续跑（脚本自带断点续传）。
另外监控 E: 剩余空间与文件总量，写日志 watchdog.log。
用户离开期间无人值守时保命用。"""
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = r"E:\binance\klines_1m"
LOG = os.path.join(OUT, "watchdog.log")
PY = sys.executable
CHECK_SEC = 120


def log(s):
    line = "%s  %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), s)
    print(line, flush=True)
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def fetch_running():
    """wmic 在新版 Windows 已移除（实测恒失败），改用 PowerShell Get-CimInstance。"""
    ps = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
          "ForEach-Object { $_.CommandLine }")
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", ps],
            stderr=subprocess.DEVNULL, timeout=60).decode("utf-8", "ignore")
        return "fetch_1m_all.py" in out
    except Exception as e:
        log("进程检查失败(%r)，按在跑处理以免重复启动" % (e,))
        return True


def disk_free_gb(path):
    try:
        return shutil.disk_usage(path).free / 1e9
    except Exception:
        return -1


def total_gb():
    t = 0
    d = os.path.join(OUT, "1m")
    if os.path.isdir(d):
        for n in os.listdir(d):
            try:
                t += os.path.getsize(os.path.join(d, n))
            except Exception:
                pass
    return t / 1e9


def main():
    log("watchdog 启动：每 %ds 检查 fetch_1m_all.py" % CHECK_SEC)
    while True:
        free = disk_free_gb(OUT)
        tot = total_gb()
        if 0 <= free < 30:
            log("E: 剩余仅 %.1f GB，停止守护（不再启动新拉取）" % free)
            return
        if fetch_running():
            log("在跑 ok  free=%.0fGB  data=%.2fGB" % (free, tot))
        else:
            if not net_alive():
                log("未在跑，且网络不可达 -> 暂不拉起，等下一轮  free=%.0fGB  data=%.2fGB" % (free, tot))
                time.sleep(CHECK_SEC)
                continue
            log("未在跑 -> 拉起续传  free=%.0fGB  data=%.2fGB" % (free, tot))
            try:
                subprocess.run([PY, "-X", "utf8", os.path.join(HERE, "fetch_1m_all.py"),
                                "--out", OUT, "--workers", "8"], cwd=HERE)
                log("拉取进程退出（正常结束或异常），下一轮再检查")
            except Exception as e:
                log("拉起失败 %r" % (e,))
        time.sleep(CHECK_SEC)


if __name__ == "__main__":
    main()
