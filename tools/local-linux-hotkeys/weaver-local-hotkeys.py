#!/usr/bin/env python3
"""Passive X11 hotkeys for local Weaver capture.

Hotkeys:
- Shift+S: screenshot
- Shift+R+E: toggle screen recording

Uses the XRecord extension so normal typing is not stolen from focused apps.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from Xlib import X, XK, display
from Xlib.ext import record
from Xlib.protocol import rq


HOME = Path.home()
SCREENSHOT_CMD = os.environ.get("WEAVER_SCREENSHOT_CMD", str(HOME / ".local/bin/weaver-screenshot"))
RECORD_CMD = os.environ.get("WEAVER_RECORD_CMD", str(HOME / ".local/bin/weaver-record-toggle"))


class WeaverHotkeys:
    def __init__(self) -> None:
        self.ctrl = display.Display()
        self.rec = display.Display()
        if not self.rec.has_extension("RECORD"):
            raise RuntimeError("XRecord extension is not available on this X11 session")

        self.key_s = self.keycode("s")
        self.key_r = self.keycode("r")
        self.key_e = self.keycode("e")
        self.shift_keys = {self.keycode("Shift_L"), self.keycode("Shift_R")}
        self.pressed: set[int] = set()
        self.last_screenshot = 0.0
        self.last_record = 0.0
        self.debug = os.environ.get("WEAVER_HOTKEY_DEBUG") == "1"

    def keycode(self, name: str) -> int:
        code = self.ctrl.keysym_to_keycode(XK.string_to_keysym(name))
        if not code:
            raise RuntimeError(f"Could not resolve keycode for {name}")
        return code

    def shifted(self, event) -> bool:
        return bool(event.state & X.ShiftMask) or bool(self.pressed & self.shift_keys)

    def spawn(self, command: str) -> None:
        if not Path(command).exists() and not shutil.which(command):
            print(f"[weaver-hotkeys] missing command: {command}", file=sys.stderr, flush=True)
            return
        subprocess.Popen([command], start_new_session=True)

    def on_keypress(self, event) -> None:
        now = time.monotonic()
        self.pressed.add(event.detail)
        if self.debug:
            print(
                f"[weaver-hotkeys] press code={event.detail} state={event.state} "
                f"shifted={self.shifted(event)} pressed={sorted(self.pressed)}",
                flush=True,
            )

        if event.detail == self.key_s and self.shifted(event) and now - self.last_screenshot > 0.7:
            self.last_screenshot = now
            self.spawn(SCREENSHOT_CMD)
            return

        if (
            self.shifted(event)
            and self.key_r in self.pressed
            and self.key_e in self.pressed
            and now - self.last_record > 0.9
        ):
            self.last_record = now
            self.spawn(RECORD_CMD)

    def on_keyrelease(self, event) -> None:
        if self.debug:
            print(f"[weaver-hotkeys] release code={event.detail} state={event.state}", flush=True)
        self.pressed.discard(event.detail)

    def callback(self, reply) -> None:
        if reply.category != record.FromServer or reply.client_swapped or not reply.data:
            return
        data = reply.data
        while len(data):
            event, data = rq.EventField(None).parse_binary_value(data, self.ctrl.display, None, None)
            if event.type == X.KeyPress:
                self.on_keypress(event)
            elif event.type == X.KeyRelease:
                self.on_keyrelease(event)

    def check(self) -> None:
        print("Weaver local hotkeys check")
        print(f"  display: {os.environ.get('DISPLAY', '')}")
        print("  XRecord: OK")
        print(f"  Shift+S screenshot command: {SCREENSHOT_CMD}")
        print(f"  Shift+R+E record command: {RECORD_CMD}")
        print(f"  keycodes: S={self.key_s} R={self.key_r} E={self.key_e} Shift={sorted(self.shift_keys)}")

    def run(self) -> None:
        ctx = self.rec.record_create_context(
            0,
            [record.AllClients],
            [{
                "core_requests": (0, 0),
                "core_replies": (0, 0),
                "ext_requests": (0, 0, 0, 0),
                "ext_replies": (0, 0, 0, 0),
                "delivered_events": (0, 0),
                "device_events": (X.KeyPress, X.KeyRelease),
                "errors": (0, 0),
                "client_started": False,
                "client_died": False,
            }],
        )
        print("[weaver-hotkeys] listening: Shift+S screenshot, Shift+R+E record", flush=True)
        try:
            self.rec.record_enable_context(ctx, self.callback)
        finally:
            self.rec.record_free_context(ctx)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="check XRecord and keycodes, then exit")
    args = parser.parse_args()
    hotkeys = WeaverHotkeys()
    if args.check:
      hotkeys.check()
      return 0
    hotkeys.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
