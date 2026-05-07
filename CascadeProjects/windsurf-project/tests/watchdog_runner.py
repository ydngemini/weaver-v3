"""
watchdog_runner.py
──────────────────
Launches all three Weaver test suites sequentially, monitors them with
a file-system watchdog (watchdog lib) + psutil resource alerts, streams
live output, and writes a unified timestamped log.

Usage:
    venv/bin/python3 watchdog_runner.py              # full run (~35 min)
    venv/bin/python3 watchdog_runner.py --quick      # quick sampler (~5 min)
    venv/bin/python3 watchdog_runner.py --suites 1 2 # pick suites: 1=codebase 2=v5 3=30min

Suites:
    1  whole_codebase_tests.py ALL       (existing suite)
    2  stress_tests_v5.py all            (20 soul-binding tests)
    3  stress_30min_full.py              (30-min endurance)
"""

import argparse
import datetime
import os
import queue
import re
import subprocess
import sys
import threading
import time

import psutil
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.abspath(__file__))
PYTHON = os.path.join(BASE, "venv", "bin", "python3")
TS     = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG    = os.path.join(BASE, f"watchdog_run_{TS}.log")

# ── ANSI colours ──────────────────────────────────────────────────────────────
RST  = "\033[0m"
BOLD = "\033[1m"
RED  = "\033[31m"
GRN  = "\033[32m"
YLW  = "\033[33m"
BLU  = "\033[34m"
MAG  = "\033[35m"
CYN  = "\033[36m"
WHT  = "\033[37m"

# ── Suite definitions ─────────────────────────────────────────────────────────
SUITES = {
    1: {
        "name":    "Codebase Tests",
        "cmd":     [PYTHON, "whole_codebase_tests.py", "ALL"],
        "colour":  CYN,
        "timeout": 600,          # 10 min hard cap
    },
    2: {
        "name":    "Stress Tests v5 (soul-binding)",
        "cmd":     [PYTHON, "stress_tests_v5.py", "all"],
        "colour":  MAG,
        "timeout": 300,          # 5 min
    },
    3: {
        "name":    "30-min Full Endurance",
        "cmd_full":  [PYTHON, "stress_30min_full.py"],
        "cmd_quick": [PYTHON, "stress_30min_full.py", "--quick"],
        "colour":  YLW,
        "timeout": 2100,         # 35 min
        "timeout_quick": 300,    # 5 min
    },
}

# ── Global counters ───────────────────────────────────────────────────────────
global_pass = 0
global_fail = 0
global_error = 0
_counter_lock = threading.Lock()

# ── Pattern matchers ──────────────────────────────────────────────────────────
_PAT_PASS  = re.compile(r"\b(PASS|OK|passed|✓)\b",  re.IGNORECASE)
_PAT_FAIL  = re.compile(r"\b(FAIL|FAILED|FAILURE)\b", re.IGNORECASE)
_PAT_ERR   = re.compile(r"\b(ERROR|EXCEPTION|Traceback)\b", re.IGNORECASE)
_PAT_WARN  = re.compile(r"\b(WARN|WARNING|SKIP)\b",  re.IGNORECASE)

# ── Log file handle ───────────────────────────────────────────────────────────
_log_fh = None

def _log_open():
    global _log_fh
    _log_fh = open(LOG, "w", buffering=1)
    _log_fh.write(f"=== Weaver Watchdog Run — {TS} ===\n\n")

def _log(line: str):
    if _log_fh:
        _log_fh.write(line + "\n")
        _log_fh.flush()

def _log_close():
    if _log_fh:
        _log_fh.close()

# ── Pretty print ──────────────────────────────────────────────────────────────
def _stamp() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S")

def _print(prefix: str, colour: str, line: str):
    """Colour-code, count, and print a line."""
    global global_pass, global_fail, global_error
    tag = ""
    if _PAT_PASS.search(line):
        tag = f"{GRN}[PASS]{RST} "
        with _counter_lock:
            global_pass += 1
    elif _PAT_FAIL.search(line):
        tag = f"{RED}[FAIL]{RST} "
        with _counter_lock:
            global_fail += 1
    elif _PAT_ERR.search(line):
        tag = f"{RED}[ERR ]{RST} "
        with _counter_lock:
            global_error += 1
    elif _PAT_WARN.search(line):
        tag = f"{YLW}[WARN]{RST} "

    msg = f"{BOLD}{colour}[{prefix}]{RST} {_stamp()} {tag}{line}"
    print(msg)
    _log(f"[{prefix}] {_stamp()} {line}")

def _section(title: str):
    bar = "─" * (72 - len(title) - 2)
    print(f"\n{BOLD}{BLU}── {title} {bar}{RST}")
    _log(f"\n── {title} {'─'*(72-len(title)-2)}")

# ── Resource monitor thread ───────────────────────────────────────────────────
_res_stop = threading.Event()

def _resource_monitor(interval: int = 30):
    """Print CPU/RAM/Disk every `interval` seconds; alert on high usage."""
    while not _res_stop.is_set():
        time.sleep(interval)
        if _res_stop.is_set():
            break
        cpu  = psutil.cpu_percent(interval=1)
        ram  = psutil.virtual_memory()
        disk = psutil.disk_usage(BASE)

        ram_pct  = ram.percent
        disk_pct = disk.percent
        ram_gb   = ram.used / 1e9
        ram_tot  = ram.total / 1e9

        colour = GRN
        if cpu > 90 or ram_pct > 85:
            colour = RED
        elif cpu > 70 or ram_pct > 70:
            colour = YLW

        msg = (f"CPU {cpu:5.1f}%  RAM {ram_gb:.1f}/{ram_tot:.1f}GB ({ram_pct:.0f}%)"
               f"  Disk {disk_pct:.0f}%")
        print(f"{colour}[WDG] {_stamp()} [SYS] {msg}{RST}")
        _log(f"[WDG] {_stamp()} [SYS] {msg}")

        if ram_pct > 85:
            print(f"{RED}{BOLD}[WDG] ⚠  MEMORY PRESSURE — {ram_pct:.0f}% used{RST}")
            _log(f"[WDG] ALERT: MEMORY PRESSURE {ram_pct:.0f}%")

# ── File watchdog (tails LOG for new FAIL/ERROR lines) ───────────────────────
_alert_q: queue.Queue = queue.Queue()

class _LogWatcher(FileSystemEventHandler):
    def __init__(self, path):
        self._path = os.path.abspath(path)
        self._pos  = 0

    def on_modified(self, event):
        if os.path.abspath(event.src_path) != self._path:
            return
        try:
            with open(self._path) as f:
                f.seek(self._pos)
                for line in f:
                    if _PAT_FAIL.search(line) or _PAT_ERR.search(line):
                        _alert_q.put(line.rstrip())
                self._pos = f.tell()
        except Exception:
            pass

def _alert_printer():
    while True:
        try:
            line = _alert_q.get(timeout=1)
        except queue.Empty:
            continue
        if line is None:
            break
        print(f"{RED}{BOLD}[WDG] !! LOG ALERT: {line}{RST}")

# ── Process runner ────────────────────────────────────────────────────────────
def _stream_proc(proc, prefix, colour):
    """Stream stdout/stderr from proc line-by-line."""
    def _reader(stream, label):
        for raw in stream:
            line = raw.rstrip()
            if line:
                _print(f"{prefix}/{label}", colour, line)
    t_out = threading.Thread(target=_reader, args=(proc.stdout, "OUT"), daemon=True)
    t_err = threading.Thread(target=_reader, args=(proc.stderr, "ERR"), daemon=True)
    t_out.start(); t_err.start()
    t_out.join();  t_err.join()

def run_suite(suite_id: int, quick: bool = False) -> int:
    """Run a single suite. Returns exit code."""
    cfg = SUITES[suite_id]
    name = cfg["name"]
    colour = cfg["colour"]

    if suite_id == 3:
        cmd     = cfg["cmd_quick"] if quick else cfg["cmd_full"]
        timeout = cfg["timeout_quick"] if quick else cfg["timeout"]
    else:
        cmd     = cfg["cmd"]
        timeout = cfg["timeout"]

    _section(f"Suite {suite_id}: {name}")
    print(f"{colour}[WDG] {_stamp()} Launching: {' '.join(cmd)}{RST}")
    _log(f"[WDG] Launching suite {suite_id}: {' '.join(cmd)}")

    start = time.time()
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=BASE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        stream_thread = threading.Thread(
            target=_stream_proc, args=(proc, f"S{suite_id}", colour), daemon=True
        )
        stream_thread.start()

        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            print(f"{RED}[WDG] {_stamp()} Suite {suite_id} TIMED OUT after {timeout}s — killed{RST}")
            _log(f"[WDG] Suite {suite_id} TIMEOUT after {timeout}s")
            stream_thread.join(timeout=5)
            return -1

        stream_thread.join(timeout=10)
        elapsed = time.time() - start
        code = proc.returncode
        status = f"{GRN}EXIT 0 (ok){RST}" if code == 0 else f"{RED}EXIT {code} (FAILED){RST}"
        print(f"{BOLD}[WDG] {_stamp()} Suite {suite_id} done in {elapsed:.1f}s — {status}{RST}")
        _log(f"[WDG] Suite {suite_id} done in {elapsed:.1f}s — exit {code}")
        return code

    except FileNotFoundError as exc:
        print(f"{RED}[WDG] {_stamp()} Suite {suite_id} could not start: {exc}{RST}")
        _log(f"[WDG] Suite {suite_id} could not start: {exc}")
        return -2

# ── Weaver component health check ─────────────────────────────────────────────
def _check_weaver_components():
    """Quick import-based health check of core Weaver modules."""
    _section("Weaver Component Health Check")
    components = [
        ("akashic_hub",    "AkashicHub"),
        ("liquid_fracture","LiquidFractureEngine"),
        ("quantum_soul",   "build_fracture_circuit"),
        ("quantum_networks","QuantumNetworkOrchestrator"),
        ("nexus_bus",      None),
    ]
    all_ok = True
    for module, symbol in components:
        try:
            result = subprocess.run(
                [PYTHON, "-c",
                 f"import {module}" + (f"; from {module} import {symbol}" if symbol else "")],
                capture_output=True, text=True, cwd=BASE, timeout=15,
            )
            ok = result.returncode == 0
            tag = f"{GRN}OK{RST}" if ok else f"{RED}FAIL{RST}"
            detail = "" if ok else f" — {result.stderr.strip()[:120]}"
            print(f"  {tag}  {module}.{symbol or '(import)'}{detail}")
            _log(f"  {'OK' if ok else 'FAIL'}  {module}.{symbol or '(import)'}{detail}")
            if not ok:
                all_ok = False
        except subprocess.TimeoutExpired:
            print(f"  {RED}TIMEOUT{RST}  {module}")
            _log(f"  TIMEOUT  {module}")
            all_ok = False
    return all_ok

# ── Final summary ─────────────────────────────────────────────────────────────
def _summary(results: dict, total_elapsed: float):
    _section("WATCHDOG FINAL SUMMARY")
    print(f"  Total wall time : {total_elapsed/60:.1f} min")
    print(f"  {GRN}PASS lines      : {global_pass}{RST}")
    print(f"  {RED}FAIL lines      : {global_fail}{RST}")
    print(f"  {RED}ERROR lines     : {global_error}{RST}")
    print()
    for sid, code in results.items():
        name = SUITES[sid]["name"]
        if code == 0:
            mark = f"{GRN}✓ PASSED{RST}"
        elif code == -1:
            mark = f"{YLW}⏱ TIMEOUT{RST}"
        elif code == -2:
            mark = f"{RED}✗ NOT FOUND{RST}"
        else:
            mark = f"{RED}✗ FAILED (exit {code}){RST}"
        print(f"  Suite {sid}: {name:<40} {mark}")

    overall = all(c == 0 for c in results.values())
    banner = f"{GRN}{BOLD}ALL SUITES PASSED{RST}" if overall else f"{RED}{BOLD}SOME SUITES FAILED{RST}"
    print(f"\n  {banner}")
    print(f"  Log: {LOG}\n")
    _log(f"\nSUMMARY: pass={global_pass} fail={global_fail} error={global_error}")
    _log(f"Overall: {'PASS' if overall else 'FAIL'}")

# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Weaver watchdog test runner")
    parser.add_argument("--quick",  action="store_true",
                        help="Run suite 3 in --quick mode (~3 min instead of 30)")
    parser.add_argument("--suites", nargs="+", type=int, choices=[1, 2, 3],
                        default=[1, 2, 3], metavar="N",
                        help="Which suites to run (1=codebase 2=v5 3=30min)")
    args = parser.parse_args()

    _log_open()

    print(f"\n{BOLD}{BLU}{'═'*72}{RST}")
    print(f"{BOLD}{BLU}  Weaver Watchdog Runner — {TS}{RST}")
    print(f"{BOLD}{BLU}  Suites: {args.suites}{'  [QUICK]' if args.quick else ''}{RST}")
    print(f"{BOLD}{BLU}  Log   : {LOG}{RST}")
    print(f"{BOLD}{BLU}{'═'*72}{RST}\n")

    # ── Component health ───────────────────────────────────────────────────────
    _check_weaver_components()

    # ── File watchdog (monitors log for FAIL/ERROR) ────────────────────────────
    watcher  = _LogWatcher(LOG)
    observer = Observer()
    observer.schedule(watcher, path=os.path.dirname(LOG), recursive=False)
    observer.start()
    alert_thread = threading.Thread(target=_alert_printer, daemon=True)
    alert_thread.start()

    # ── Resource monitor ───────────────────────────────────────────────────────
    res_thread = threading.Thread(target=_resource_monitor, args=(30,), daemon=True)
    res_thread.start()

    # ── Run suites ─────────────────────────────────────────────────────────────
    results = {}
    wall_start = time.time()

    for sid in args.suites:
        results[sid] = run_suite(sid, quick=args.quick)

    total_elapsed = time.time() - wall_start

    # ── Teardown ───────────────────────────────────────────────────────────────
    _res_stop.set()
    _alert_q.put(None)
    observer.stop()
    observer.join(timeout=5)

    # ── Summary ────────────────────────────────────────────────────────────────
    _summary(results, total_elapsed)
    _log_close()

    # Exit non-zero if any suite failed
    sys.exit(0 if all(c == 0 for c in results.values()) else 1)


if __name__ == "__main__":
    main()
