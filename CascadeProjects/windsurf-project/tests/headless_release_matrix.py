#!/usr/bin/env python3
"""Five-viewport, mocked-cortex release gate for Weaver's public headless shell."""

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone

from playwright.async_api import async_playwright


URL = os.environ.get(
    "WEAVER_HEADLESS_AUDIT_URL",
    "http://127.0.0.1:8018/headless.html",
)
CSRF = "C" * 32
IPHONE_UA = (
    "Mozilla/5.0 (iPhone; CPU iPhone OS 18_5 like Mac OS X) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.5 Mobile/15E148 Safari/604.1"
)
DESKTOP_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0 Safari/537.36"
)
VIEWPORTS = (
    {"name": "compact-320", "width": 320, "height": 568, "dpr": 2, "mobile": True},
    {"name": "iphone-16e", "width": 390, "height": 844, "dpr": 3, "mobile": True},
    {
        "name": "tablet-768",
        "width": 768,
        "height": 1024,
        "dpr": 2,
        "mobile": False,
        "touch": True,
    },
    {"name": "desktop-1024", "width": 1024, "height": 768, "dpr": 1, "mobile": False},
    {"name": "desktop-1440", "width": 1440, "height": 900, "dpr": 1, "mobile": False},
)


def snapshot(revision=100):
    return {
        "schema_version": 2,
        "revision": revision,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "freshness": {
            name: {"fresh": name != "camera"}
            for name in (
                "headless", "fabric", "cognition", "dependencies", "body",
                "environment", "camera", "microphone",
            )
        },
        "system": {
            "active": True,
            "ready": True,
            "status": "nominal",
            "uptime_seconds": 600,
        },
        "awareness": {
            "status": "limited",
            "confidence": 0.88,
            "degraded_reasons": ["camera-stale"],
        },
        "voice": {
            "status": "ready",
            "transport": {
                "device": {
                    "device_class": "iphone-16e",
                    "thermal_state": "nominal",
                    "low_power_mode": False,
                }
            },
        },
        "cognition": {
            "status": "nominal",
            "phase": "idle",
            "thought_count": 9,
            "dream_count": 2,
            "private_thought_available": True,
            "private_dream_available": True,
            "thought_topics": ["release"],
            "dream_topics": ["continuity"],
        },
        "fabric": {
            "status": "nominal",
            "pressure": 0.18,
            "ledger_valid": True,
            "ledger_sequence": 88,
            "lanes": {
                name: {"active": 0, "queued": 0}
                for name in ("realtime", "interactive", "embodiment", "background")
            },
        },
    }


def init_script(initial):
    payload = json.dumps(initial, separators=(",", ":"))
    return f"""
sessionStorage.setItem('weaver_llm_key', 'matrix-release-key');
Object.defineProperty(window, 'WebGLRenderingContext', {{value: undefined, configurable: true}});
Object.defineProperty(window, 'WebGL2RenderingContext', {{value: undefined, configurable: true}});
window.__matrixSnapshot = {payload};
window.__matrixSockets = [];
window.__matrixOutbound = [];
window.__matrixProtocols = [];
window.__matrixConnectedAt = null;
class MatrixSocket extends EventTarget {{
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;
  constructor(url, protocols) {{
    super();
    this.url = url;
    this.protocols = Array.isArray(protocols) ? protocols : [protocols];
    this.readyState = MatrixSocket.CONNECTING;
    window.__matrixSockets.push(this);
    window.__matrixProtocols.push([url, ...this.protocols]);
    setTimeout(() => {{
      if (this.readyState !== MatrixSocket.CONNECTING) return;
      this.readyState = MatrixSocket.OPEN;
      window.__matrixConnectedAt = performance.now();
      this.dispatchEvent(new Event('open'));
      this.emit({{type:'hello',schema_version:2,correlation_id:'matrix.state.1',heartbeat_interval_ms:10000,revision:window.__matrixSnapshot.revision}});
      this.emit({{type:'snapshot',snapshot:structuredClone(window.__matrixSnapshot)}});
    }}, 15);
  }}
  send(raw) {{ window.__matrixOutbound.push(JSON.parse(raw)); }}
  emit(value) {{ this.dispatchEvent(new MessageEvent('message', {{data: JSON.stringify(value)}})); }}
  close(code=1000, reason='') {{
    if (this.readyState === MatrixSocket.CLOSED) return;
    this.readyState = MatrixSocket.CLOSED;
    this.dispatchEvent(new CloseEvent('close', {{code, reason, wasClean: code === 1000}}));
  }}
}}
window.WebSocket = MatrixSocket;
"""


async def install_routes(context, initial):
    async def session(route):
        await route.fulfill(
            status=200,
            content_type="application/json",
            headers={
                "set-cookie": (
                    "weaver_headless_session=matrix; HttpOnly; Secure; "
                    "SameSite=Strict; Path=/"
                ),
                "cache-control": "no-store",
            },
            body=json.dumps({
                "schema_version": 2,
                "csrf_token": CSRF,
                "expires_at": (
                    datetime.now(timezone.utc) + timedelta(seconds=600)
                ).isoformat(),
                "expires_in_seconds": 600,
            }),
        )

    async def state(route):
        await route.fulfill(
            status=200,
            content_type="application/json",
            headers={"cache-control": "no-store"},
            body=json.dumps(initial),
        )

    await context.route("**/brain/headless/v2/session", session)
    await context.route("**/brain/headless/v2/state", state)


async def audit_viewport(browser, spec):
    initial = snapshot()
    context = await browser.new_context(
        viewport={"width": spec["width"], "height": spec["height"]},
        screen={"width": spec["width"], "height": spec["height"]},
        device_scale_factor=spec["dpr"],
        is_mobile=spec["mobile"],
        has_touch=spec.get("touch", spec["mobile"]),
        user_agent=IPHONE_UA if spec["mobile"] else DESKTOP_UA,
    )
    await context.add_init_script(init_script(initial))
    await install_routes(context, initial)
    page = await context.new_page()
    errors = []
    page.on("pageerror", lambda error: errors.append(f"page:{error}"))
    page.on(
        "console",
        lambda message: errors.append(f"console:{message.text}")
        if message.type == "error" else None,
    )
    wall_start = time.perf_counter()
    await page.goto(URL, wait_until="domcontentloaded")
    await page.wait_for_function(
        "() => globalThis.__weaverHeadlessVisualAudit?.().ready === true",
        timeout=5_000,
    )
    await page.wait_for_function(
        "() => globalThis.__weaverHeadlessStateChannelAudit?.().status === 'connected'",
        timeout=5_000,
    )
    ready_wall_ms = round((time.perf_counter() - wall_start) * 1_000, 2)
    await page.evaluate("""() => window.__matrixSockets.at(-1).emit({
      type:'heartbeat',sent_at:new Date().toISOString(),revision:window.__matrixSnapshot.revision
    })""")
    await page.wait_for_timeout(120)

    metrics = await page.evaluate("""() => {
      const visual = globalThis.__weaverHeadlessVisualAudit();
      const channel = globalThis.__weaverHeadlessStateChannelAudit();
      const lifecycle = globalThis.__weaverHeadlessLifecycleAudit();
      const resources = performance.getEntriesByType('resource');
      const visibleButtons = [...document.querySelectorAll('button')].filter(button => {
        const style = getComputedStyle(button);
        const rect = button.getBoundingClientRect();
        return style.display !== 'none' && style.visibility !== 'hidden' && !button.hidden
          && rect.width > 0 && rect.height > 0;
      });
      const rects = ['conversation', 'awarenessTitle', 'bar', 'diagnosticsToggle'].map(id => {
        const rect = document.getElementById(id).getBoundingClientRect();
        return {id, left: rect.left, right: rect.right, width: rect.width};
      });
      return {
        inner: {width: innerWidth, height: innerHeight},
        scroll: {
          html: document.documentElement.scrollWidth,
          body: document.body.scrollWidth,
        },
        visual,
        channel,
        lifecycle,
        connectedAtMs: window.__matrixConnectedAt,
        resourceCount: resources.length,
        heapBytes: performance.memory?.usedJSHeapSize ?? null,
        buttonSizes: visibleButtons.map(button => {
          const rect = button.getBoundingClientRect();
          return {
            id: button.id,
            width: rect.width,
            height: rect.height,
            minimum: Math.min(rect.width, rect.height),
          };
        }).sort((a, b) => a.minimum - b.minimum),
        unlabeledButtons: visibleButtons.filter(button =>
          !button.textContent.trim() && !button.getAttribute('aria-label')).length,
        duplicateIds: (() => {
          const ids = [...document.querySelectorAll('[id]')].map(element => element.id);
          return ids.filter((id, index) => ids.indexOf(id) !== index);
        })(),
        rects,
        protocols: window.__matrixProtocols,
        outbound: window.__matrixOutbound,
      };
    }""")

    await page.locator("#diagnosticsToggle").click()
    await page.wait_for_function(
        "() => document.activeElement?.id === 'diagnosticsClose'"
    )
    await page.wait_for_timeout(350)
    dialog = await page.evaluate("""() => {
      const drawer = document.getElementById('diagnosticsDrawer');
      const rect = drawer.getBoundingClientRect();
      return {
        role: drawer.getAttribute('role'),
        modal: drawer.getAttribute('aria-modal'),
        left: rect.left,
        right: rect.right,
        openAudit: globalThis.__weaverHeadlessAccessibilityAudit(),
      };
    }""")
    await page.keyboard.press("Escape")
    await page.screenshot(
        path=f"/tmp/weaver-headless-{spec['name']}.png",
        full_page=False,
    )

    if spec["name"] == "iphone-16e":
        await context.set_offline(True)
        await page.wait_for_function(
            "() => globalThis.__weaverHeadlessStateChannelAudit().status === 'paused'"
        )
        await context.set_offline(False)
        await page.wait_for_function(
            "() => globalThis.__weaverHeadlessStateChannelAudit().status === 'connected'",
            timeout=5_000,
        )
        metrics["offlineRecovered"] = True

    result = {
        **spec,
        "readyWallMs": ready_wall_ms,
        "metrics": metrics,
        "dialog": dialog,
        "errors": errors,
    }
    assert metrics["scroll"]["html"] <= metrics["inner"]["width"] + 1, result
    assert metrics["scroll"]["body"] <= metrics["inner"]["width"] + 1, result
    assert all(
        rect["left"] >= -1 and rect["right"] <= spec["width"] + 1
        for rect in metrics["rects"]
    ), result
    assert metrics["visual"]["ready"] and metrics["visual"]["fallback2d"], result
    assert metrics["visual"]["semantic"]["verified"] is True, result
    assert metrics["visual"]["semantic"]["revision"] == 100, result
    assert metrics["channel"]["authenticated"] is True, result
    assert metrics["channel"]["canExecuteCapsules"] is False, result
    assert metrics["channel"]["maxSemanticWaitMs"] is None, result
    assert metrics["channel"]["heartbeatAgeMs"] is not None, result
    assert all(item.get("type") == "resume" for item in metrics["outbound"]), result
    assert metrics["protocols"][0][1:] == [
        "weaver-headless-v2",
        f"weaver-csrf.{CSRF}",
    ], result
    assert metrics["unlabeledButtons"] == 0 and not metrics["duplicateIds"], result
    minimum_target = 43.5 if spec.get("touch", spec["mobile"]) else 23.5
    assert metrics["buttonSizes"][0]["minimum"] >= minimum_target, result
    assert dialog["role"] == "dialog" and dialog["modal"] == "true", result
    assert dialog["left"] >= -1 and dialog["right"] <= spec["width"] + 1, result
    assert dialog["openAudit"]["backgroundInert"] is True, result
    assert ready_wall_ms < 2_500, result
    assert metrics["connectedAtMs"] is not None and metrics["connectedAtMs"] < 2_000, result
    assert metrics["resourceCount"] <= 24, result
    if metrics["heapBytes"] is not None:
        assert metrics["heapBytes"] < 96 * 1024 * 1024, result
    if spec["name"] == "iphone-16e":
        assert metrics["visual"]["iPhone16e"] is True, result
        assert metrics["visual"]["profile"] == "iphone-16e-adaptive", result
        assert metrics["visual"]["fps"] == 60, result
        assert metrics["visual"]["dprCap"] == 1.25, result
        assert metrics["visual"]["effectiveDpr"] <= 1.25, result
        assert metrics.get("offlineRecovered") is True, result
    assert not errors, result
    await context.close()
    return result


async def main():
    selected = set(sys.argv[1:])
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        results = []
        for viewport in VIEWPORTS:
            if selected and viewport["name"] not in selected:
                continue
            results.append(await audit_viewport(browser, viewport))
        await browser.close()
    summary = {
        "viewports": [
            {
                "name": item["name"],
                "size": f"{item['width']}x{item['height']}",
                "readyWallMs": item["readyWallMs"],
                "profile": item["metrics"]["visual"]["profile"],
                "fps": item["metrics"]["visual"]["fps"],
                "effectiveDpr": item["metrics"]["visual"]["effectiveDpr"],
                "resources": item["metrics"]["resourceCount"],
                "heapMiB": round(
                    (item["metrics"]["heapBytes"] or 0) / 1024 / 1024,
                    2,
                ),
                "horizontalOverflow": max(
                    item["metrics"]["scroll"]["html"],
                    item["metrics"]["scroll"]["body"],
                ) - item["metrics"]["inner"]["width"],
                "errors": item["errors"],
            }
            for item in results
        ],
        "iphoneOfflineRecovered": next(
            item["metrics"].get("offlineRecovered", False)
            for item in results if item["name"] == "iphone-16e"
        ) if any(item["name"] == "iphone-16e" for item in results) else None,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(main())
