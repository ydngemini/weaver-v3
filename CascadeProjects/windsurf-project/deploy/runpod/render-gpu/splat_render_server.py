#!/usr/bin/env python3
"""RunPod CUDA splat renderer for Weaver.

This is intentionally dependency-light: stdlib HTTP, Pillow for JPEG encoding,
and torch for CUDA work. It renders a live 3D Gaussian-style penthouse/splat
scene on the pod GPU and streams it as MJPEG for the browser to composite
behind the local avatar.
"""

from __future__ import annotations

import io
import json
import math
import os
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import torch
from PIL import Image


HOST = os.environ.get("WEAVER_RENDER_HOST", "0.0.0.0")
PORT = int(os.environ.get("WEAVER_RENDER_PORT", "8888"))
DEFAULT_W = int(os.environ.get("WEAVER_RENDER_W", "1280"))
DEFAULT_H = int(os.environ.get("WEAVER_RENDER_H", "720"))
DEFAULT_FPS = int(os.environ.get("WEAVER_RENDER_FPS", "60"))
POINTS = int(os.environ.get("WEAVER_RENDER_POINTS", "22000"))


class CudaSplatRenderer:
    def __init__(self, width: int = DEFAULT_W, height: int = DEFAULT_H, points: int = POINTS):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available; refusing to serve a fake GPU renderer")
        self.device = torch.device("cuda")
        self.width = width
        self.height = height
        self.points = points
        self.last_ms = 0.0
        self.last_encode_ms = 0.0
        self.frames = 0
        self.started = time.time()
        self._base_cache: dict[tuple[int, int], torch.Tensor] = {}
        self._build_scene(points)

    def _build_scene(self, points: int) -> None:
        g = torch.Generator(device=self.device)
        g.manual_seed(404870839825)
        target = max(8500, min(int(points), 26000))
        parts: list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []

        def rand(n: int) -> torch.Tensor:
            return torch.rand((n,), generator=g, device=self.device)

        def colorize(base: tuple[float, float, float], n: int, noise: float = 0.05) -> torch.Tensor:
            c = torch.tensor(base, device=self.device).view(1, 3).repeat(n, 1)
            if noise:
                c += (torch.rand((n, 3), generator=g, device=self.device) - 0.5) * noise
            return torch.clamp(c, 0.0, 1.8)

        def add(
            name: str,
            xyz: torch.Tensor,
            color: torch.Tensor,
            size: torch.Tensor | float,
            alpha: torch.Tensor | float,
            *,
            shimmer: float = 0.0,
            speed: tuple[float, float] = (0.08, 0.28),
            spike: tuple[float, float] = (18.0, 30.0),
        ) -> None:
            n = xyz.shape[0]
            if not isinstance(size, torch.Tensor):
                size_t = torch.full((n,), float(size), device=self.device)
            else:
                size_t = size
            if not isinstance(alpha, torch.Tensor):
                alpha_t = torch.full((n,), float(alpha), device=self.device)
            else:
                alpha_t = alpha
            parts.append((
                name,
                xyz,
                color,
                size_t,
                alpha_t,
                torch.rand((n,), generator=g, device=self.device) * (math.pi * 2.0),
                torch.full((n,), speed[0], device=self.device) + rand(n) * (speed[1] - speed[0]),
                torch.full((n,), spike[0], device=self.device) + rand(n) * (spike[1] - spike[0]),
            ))

        def box_points(
            n: int,
            xr: tuple[float, float],
            yr: tuple[float, float],
            zr: tuple[float, float],
        ) -> torch.Tensor:
            return torch.stack([
                xr[0] + rand(n) * (xr[1] - xr[0]),
                yr[0] + rand(n) * (yr[1] - yr[0]),
                zr[0] + rand(n) * (zr[1] - zr[0]),
            ], dim=1)

        def line_points(
            n: int,
            a: tuple[float, float, float],
            b: tuple[float, float, float],
            jitter: tuple[float, float, float] = (0.015, 0.015, 0.015),
        ) -> torch.Tensor:
            u = rand(n).view(-1, 1)
            av = torch.tensor(a, device=self.device).view(1, 3)
            bv = torch.tensor(b, device=self.device).view(1, 3)
            j = (torch.rand((n, 3), generator=g, device=self.device) - 0.5)
            j *= torch.tensor(jitter, device=self.device).view(1, 3)
            return av * (1.0 - u) + bv * u + j

        # Counts are proportional so WEAVER_RENDER_POINTS can tune quality.
        def n(frac: float, minimum: int) -> int:
            return max(minimum, int(target * frac))

        # Wide marble floor, dry porcelain terrace, and reflective runway.
        floor_n = n(0.28, 2800)
        z = 1.05 + rand(floor_n) * 5.25
        span = 2.0 + z * 0.78
        x = (rand(floor_n) - 0.5) * span * 2.0
        y = -1.18 + (rand(floor_n) - 0.5) * 0.026
        floor_xyz = torch.stack([x, y, z], dim=1)
        marble = colorize((0.76, 0.65, 0.46), floor_n, 0.18)
        vein = (torch.sin(x * 3.2 + z * 1.7) > 0.86).float().view(-1, 1)
        marble = marble * (1.0 - vein * 0.32) + torch.tensor([0.95, 0.82, 0.55], device=self.device).view(1, 3) * vein * 0.22
        add("honed_marble_floor", floor_xyz, marble, 4.8 + rand(floor_n) * 7.5, 0.82, shimmer=0.035)

        refl_n = n(0.05, 520)
        refl_xyz = torch.stack([
            (rand(refl_n) - 0.5) * 8.7,
            torch.full((refl_n,), -1.155, device=self.device),
            1.35 + rand(refl_n) * 4.55,
        ], dim=1)
        refl_color = colorize((0.94, 0.68, 0.28), refl_n, 0.20)
        cool_mask = (rand(refl_n) > 0.58).float().view(-1, 1)
        refl_color = refl_color * (1.0 - cool_mask) + colorize((0.36, 0.52, 0.92), refl_n, 0.14) * cool_mask
        add("floor_city_reflection_splats", refl_xyz, refl_color, 1.4 + rand(refl_n) * 3.4, 0.12, shimmer=0.36, speed=(0.15, 0.55))

        terrace_n = n(0.06, 420)
        terrace_xyz = box_points(terrace_n, (-5.9, 5.9), (-1.17, -1.13), (5.7, 6.7))
        add("flush_porcelain_terrace", terrace_xyz, colorize((0.42, 0.37, 0.29), terrace_n, 0.08), 2.1 + rand(terrace_n) * 3.2, 0.36)

        # Curtain wall glass and bronze/thermal-break mullions.
        glass_n = n(0.07, 720)
        glass_xyz = box_points(glass_n, (-5.55, 5.55), (-0.82, 2.55), (5.72, 5.82))
        glass_color = colorize((0.13, 0.20, 0.32), glass_n, 0.10)
        add("low_e_glass_curtain_wall", glass_xyz, glass_color, 0.7 + rand(glass_n) * 1.5, 0.045, shimmer=0.08)

        line_n = max(70, target // 180)
        for i, x_pos in enumerate(torch.linspace(-5.35, 5.35, 10, device=self.device).tolist()):
            pts = line_points(line_n, (x_pos, -0.86, 5.64), (x_pos, 2.66, 5.64), (0.012, 0.018, 0.012))
            add("aged_bronze_unitized_mullion", pts, colorize((0.70, 0.48, 0.22), line_n, 0.08), 2.0, 0.62, shimmer=0.04)
            pts2 = line_points(line_n // 2, (x_pos + 0.035, -0.76, 5.60), (x_pos + 0.035, 2.48, 5.60), (0.006, 0.012, 0.006))
            add("black_polyamide_thermal_break", pts2, colorize((0.025, 0.023, 0.022), line_n // 2, 0.01), 1.65, 0.50)
        for y_pos in [-0.20, 0.80, 1.76, 2.55]:
            pts = line_points(line_n * 2, (-5.55, y_pos, 5.62), (5.55, y_pos, 5.62), (0.018, 0.010, 0.012))
            add("curtain_wall_transom", pts, colorize((0.65, 0.43, 0.19), line_n * 2, 0.08), 1.8, 0.56, shimmer=0.05)

        # Distant skyline: faint building volume plus randomized high-spike windows.
        city_n = n(0.08, 1100)
        columns = torch.randint(0, 92, (city_n,), generator=g, device=self.device).float()
        tower_seed = torch.rand((92,), generator=g, device=self.device)
        tower_h = 0.75 + torch.pow(tower_seed, 0.44) * 4.15
        tower_w = 0.08 + torch.rand((92,), generator=g, device=self.device) * 0.23
        x_center = -6.6 + (columns / 91.0) * 13.2
        x = x_center + (rand(city_n) - 0.5) * tower_w[columns.long()]
        y = -0.58 + torch.pow(rand(city_n), 1.05) * tower_h[columns.long()]
        z = 7.0 + rand(city_n) * 2.5
        city_xyz = torch.stack([x, y, z], dim=1)
        palette = torch.tensor([
            [1.00, 0.70, 0.34],
            [0.54, 0.68, 1.00],
            [1.00, 0.91, 0.64],
            [1.00, 0.48, 0.25],
            [0.70, 0.86, 1.00],
        ], device=self.device)
        city_color = palette[torch.randint(0, len(palette), (city_n,), generator=g, device=self.device)]
        add("night_city_window_gaussians", city_xyz, city_color, 0.30 + rand(city_n) * 1.15, 0.018 + rand(city_n) * 0.048, shimmer=0.92, speed=(0.18, 1.90), spike=(10, 30))

        haze_n = n(0.04, 420)
        haze_xyz = box_points(haze_n, (-6.8, 6.8), (-0.45, 2.8), (7.5, 9.7))
        add("distant_tower_haze", haze_xyz, colorize((0.06, 0.09, 0.15), haze_n, 0.05), 2.0 + rand(haze_n) * 3.5, 0.045, shimmer=0.03)

        # Left lounge: low-to-glass seating, table, and bronze object.
        sofa_n = n(0.13, 1200)
        sofa_xyz = box_points(sofa_n, (-3.80, -0.45), (-1.10, -0.30), (3.35, 5.40))
        add("low_left_lounge_sofa", sofa_xyz, colorize((0.60, 0.50, 0.35), sofa_n, 0.10), 6.0 + rand(sofa_n) * 7.8, 0.94)
        for a, b in [
            ((-3.80, -0.30, 3.35), (-0.45, -0.30, 3.35)),
            ((-3.80, -0.36, 5.40), (-0.45, -0.36, 5.40)),
            ((-3.80, -1.05, 3.35), (-3.80, -0.30, 5.40)),
        ]:
            pts = line_points(line_n * 2, a, b, (0.028, 0.028, 0.028))
            add("sofa_low_profile_edge_splats", pts, colorize((0.78, 0.62, 0.38), pts.shape[0], 0.08), 3.8, 0.70, shimmer=0.05)

        table_n = n(0.065, 680)
        table_xyz = box_points(table_n, (-1.35, 1.05), (-1.12, -0.72), (2.35, 3.65))
        add("smoked_round_coffee_table", table_xyz, colorize((0.18, 0.12, 0.06), table_n, 0.04), 5.2 + rand(table_n) * 6.8, 0.92)
        pts = line_points(line_n * 3, (-1.35, -0.70, 2.35), (1.05, -0.70, 3.65), (0.04, 0.025, 0.04))
        add("coffee_table_bronze_rim", pts, colorize((0.82, 0.55, 0.20), pts.shape[0], 0.08), 3.7, 0.74, shimmer=0.08)

        # Right island/bar: dark millwork, honed stone top, pendants.
        island_n = n(0.15, 1500)
        island_xyz = box_points(island_n, (0.95, 4.40), (-1.08, 0.36), (3.20, 5.55))
        island_color = colorize((0.12, 0.075, 0.045), island_n, 0.04)
        top_mask = (island_xyz[:, 1] > 0.05).float().view(-1, 1)
        island_color = island_color * (1.0 - top_mask) + colorize((0.70, 0.57, 0.39), island_n, 0.08) * top_mask
        add("right_waterfall_kitchen_island", island_xyz, island_color, 6.2 + rand(island_n) * 7.8, 0.96)
        for a, b in [
            ((0.95, 0.36, 3.20), (4.40, 0.36, 3.20)),
            ((0.95, 0.36, 5.55), (4.40, 0.36, 5.55)),
            ((0.95, -1.08, 3.20), (0.95, 0.36, 5.55)),
            ((4.40, -1.08, 3.20), (4.40, 0.36, 5.55)),
        ]:
            pts = line_points(line_n * 2, a, b, (0.03, 0.025, 0.03))
            add("waterfall_island_honed_edge", pts, colorize((0.88, 0.68, 0.42), pts.shape[0], 0.08), 4.1, 0.78, shimmer=0.06)

        pendant_n = max(280, target // 30)
        for x_pos in [2.65, 3.85, 5.05]:
            stem = line_points(pendant_n // 3, (x_pos, 2.65, 2.25), (x_pos, 1.55, 2.25), (0.012, 0.012, 0.012))
            add("brass_pendant_stem", stem, colorize((0.82, 0.56, 0.20), stem.shape[0], 0.08), 1.75, 0.50, shimmer=0.08)
            shade = box_points(pendant_n // 2, (x_pos - 0.22, x_pos + 0.22), (1.38, 1.78), (2.08, 2.42))
            add("warm_cylindrical_pendant_glow", shade, colorize((1.00, 0.78, 0.42), shade.shape[0], 0.12), 2.1 + rand(shade.shape[0]) * 2.3, 0.62, shimmer=0.18)

        # Cove/grazer lighting and window slot drain.
        for y_pos, z_pos, warm in [(2.58, 2.15, True), (2.50, 5.36, True), (-1.06, 5.58, False)]:
            pts = line_points(line_n * 2, (-5.6, y_pos, z_pos), (5.6, y_pos, z_pos), (0.02, 0.018, 0.018))
            color = (1.0, 0.62, 0.25) if warm else (0.50, 0.64, 0.90)
            add("knife_edge_cove_and_slot_drain", pts, colorize(color, pts.shape[0], 0.10), 2.2, 0.34, shimmer=0.12)

        self.part_counts = {}
        for name, xyz, *_ in parts:
            self.part_counts[name] = self.part_counts.get(name, 0) + int(xyz.shape[0])
        self.xyz = torch.cat([p[1] for p in parts], dim=0)
        self.color = torch.cat([p[2] for p in parts], dim=0)
        self.size = torch.cat([p[3] for p in parts], dim=0)
        self.alpha_base = torch.cat([p[4] for p in parts], dim=0)
        self.phase = torch.cat([p[5] for p in parts], dim=0)
        self.speed = torch.cat([p[6] for p in parts], dim=0)
        self.spike = torch.cat([p[7] for p in parts], dim=0)
        self.points = int(self.xyz.shape[0])

        offsets = []
        weights = []
        for yy in range(-7, 8):
            for xx in range(-7, 8):
                d2 = xx * xx + yy * yy
                offsets.append((xx, yy))
                weights.append(math.exp(-d2 / 18.0))
        self.offsets = torch.tensor(offsets, dtype=torch.long, device=self.device)
        self.weights = torch.tensor(weights, dtype=torch.float32, device=self.device)

    def resize(self, width: int, height: int) -> None:
        self.width = max(320, min(int(width), 1920))
        self.height = max(180, min(int(height), 1080))

    def _base_scene(self, w: int, h: int, t: float) -> torch.Tensor:
        """CUDA procedural penthouse base layer under the Gaussian splats."""
        cache_key = (w, h)
        cached = self._base_cache.get(cache_key)
        if cached is not None:
            return cached

        v = torch.linspace(0.0, 1.0, h, device=self.device)
        u = torch.linspace(0.0, 1.0, w, device=self.device)
        vv, uu = torch.meshgrid(v, u, indexing="ij")

        def color(rgb: tuple[float, float, float]) -> torch.Tensor:
            return torch.tensor(rgb, device=self.device).view(1, 1, 3)

        def smoothstep(a: float, b: float, x: torch.Tensor) -> torch.Tensor:
            z = torch.clamp((x - a) / max(b - a, 1e-6), 0.0, 1.0)
            return z * z * (3.0 - 2.0 * z)

        def band(x: torch.Tensor, a: float, b: float, feather: float) -> torch.Tensor:
            return smoothstep(a - feather, a + feather, x) * (1.0 - smoothstep(b - feather, b + feather, x))

        def rect(x0: float, y0: float, x1: float, y1: float, feather: float = 0.006) -> torch.Tensor:
            return band(uu, x0, x1, feather) * band(vv, y0, y1, feather)

        def ellipse(cx: float, cy: float, rx: float, ry: float, feather: float = 0.035) -> torch.Tensor:
            d = ((uu - cx) / rx) ** 2 + ((vv - cy) / ry) ** 2
            return 1.0 - smoothstep(1.0 - feather, 1.0 + feather, d)

        def line_x(x0: float, y0: float, y1: float, width: float) -> torch.Tensor:
            return rect(x0 - width, y0, x0 + width, y1, width * 0.8)

        def line_y(y0: float, x0: float, x1: float, width: float) -> torch.Tensor:
            return rect(x0, y0 - width, x1, y0 + width, width * 0.8)

        def over(dst: torch.Tensor, rgb: tuple[float, float, float] | torch.Tensor, a: torch.Tensor) -> torch.Tensor:
            c = color(rgb) if isinstance(rgb, tuple) else rgb
            aa = torch.clamp(a, 0.0, 1.0).unsqueeze(-1)
            return dst * (1.0 - aa) + c * aa

        horizon = 0.545
        img = color((0.010, 0.016, 0.030)) * (1.0 - vv[..., None]) + color((0.026, 0.024, 0.028)) * vv[..., None]

        glass = rect(0.0, 0.075, 1.0, horizon + 0.020, 0.010)
        glass_tint = color((0.026, 0.044, 0.074)) + color((0.016, 0.022, 0.030)) * vv[..., None]
        img = over(img, glass_tint, glass * 0.94)

        for x0 in [0.03, 0.12, 0.23, 0.34, 0.45, 0.59, 0.70, 0.82, 0.92]:
            width = 0.055 + 0.025 * math.sin(x0 * 31.0)
            height = 0.15 + 0.12 * abs(math.sin(x0 * 17.0))
            tower = rect(x0, horizon - height, min(0.99, x0 + width), horizon + 0.02, 0.004)
            img = over(img, (0.020, 0.026, 0.038), tower * 0.72)
            for yy in [horizon - height + 0.030, horizon - height + 0.074, horizon - height + 0.118]:
                win = line_y(yy, x0 + 0.006, min(0.99, x0 + width - 0.006), 0.0016)
                shimmer = 0.20 + 0.08 * math.sin(x0 * 19.0 + yy * 23.0)
                img = over(img, (0.95, 0.74, 0.38), win * shimmer)

        for x0 in [0.055, 0.178, 0.303, 0.428, 0.555, 0.682, 0.812, 0.942]:
            img = over(img, (0.62, 0.42, 0.18), line_x(x0, 0.075, horizon + 0.050, 0.0028) * 0.92)
            img = over(img, (0.016, 0.014, 0.012), line_x(x0 + 0.004, 0.085, horizon + 0.036, 0.0010) * 0.78)
        for y0 in [0.145, 0.292, 0.438, horizon + 0.010]:
            img = over(img, (0.54, 0.35, 0.15), line_y(y0, 0.0, 1.0, 0.0022) * 0.80)

        img = over(img, (0.090, 0.083, 0.072), rect(0.0, 0.0, 1.0, 0.115, 0.012) * 0.95)
        for y0, x0, x1 in [(0.085, 0.04, 0.48), (0.106, 0.55, 0.97)]:
            img = over(img, (1.00, 0.72, 0.36), line_y(y0, x0, x1, 0.0035) * 0.48)

        floor_a = smoothstep(horizon - 0.010, horizon + 0.018, vv)
        depth = torch.clamp((vv - horizon) / (1.0 - horizon), 0.0, 1.0)
        marble_noise = torch.sin(uu * 34.0 + vv * 19.0 + 0.2 * torch.sin(uu * 5.0)) + 0.45 * torch.sin(uu * 91.0 - vv * 37.0)
        floor_rgb = (
            color((0.33, 0.285, 0.205))
            + color((0.12, 0.09, 0.055)) * depth[..., None]
            + color((0.045, 0.035, 0.020)) * torch.clamp(marble_noise, 0.0, 1.0)[..., None]
        )
        img = over(img, floor_rgb, floor_a * 0.98)
        for x0 in [0.17, 0.36, 0.55, 0.74, 0.91]:
            grout = torch.exp(-((uu - x0) ** 2) / (0.000018 + depth * 0.00010)) * floor_a
            img = over(img, (0.18, 0.14, 0.09), grout * 0.24)
        img = over(img, (0.92, 0.64, 0.25), ellipse(0.49, 0.74, 0.22, 0.036, 0.20) * floor_a * 0.10)

        img = over(img, (0.030, 0.024, 0.018), ellipse(0.235, 0.765, 0.245, 0.060, 0.18) * 0.48)
        img = over(img, (0.42, 0.350, 0.235), rect(0.045, 0.610, 0.355, 0.700, 0.014) * 0.96)
        img = over(img, (0.285, 0.230, 0.150), rect(0.070, 0.690, 0.405, 0.800, 0.018) * 0.98)
        img = over(img, (0.52, 0.425, 0.275), line_y(0.690, 0.070, 0.405, 0.0030) * 0.70)
        for x0 in [0.16, 0.265, 0.365]:
            img = over(img, (0.23, 0.18, 0.12), line_x(x0, 0.700, 0.795, 0.0014) * 0.42)

        img = over(img, (0.025, 0.018, 0.010), ellipse(0.430, 0.830, 0.175, 0.065, 0.12) * 0.55)
        table = ellipse(0.425, 0.805, 0.145, 0.050, 0.08)
        img = over(img, (0.76, 0.50, 0.18), table * 0.66)
        img = over(img, (0.115, 0.073, 0.036), ellipse(0.425, 0.805, 0.126, 0.040, 0.08) * 0.92)

        img = over(img, (0.018, 0.013, 0.010), ellipse(0.790, 0.805, 0.250, 0.070, 0.18) * 0.60)
        img = over(img, (0.62, 0.49, 0.33), rect(0.620, 0.575, 0.980, 0.650, 0.010) * 0.98)
        img = over(img, (0.085, 0.052, 0.033), rect(0.660, 0.645, 0.995, 0.875, 0.012) * 0.98)
        img = over(img, (0.30, 0.18, 0.095), rect(0.585, 0.665, 0.668, 0.830, 0.012) * 0.90)
        img = over(img, (0.86, 0.66, 0.40), line_y(0.580, 0.620, 0.980, 0.0030) * 0.58)
        img = over(img, (0.020, 0.014, 0.010), rect(0.875, 0.650, 0.995, 0.875, 0.010) * 0.42)

        for cx in [0.690, 0.812, 0.936]:
            img = over(img, (0.76, 0.50, 0.18), line_x(cx, 0.112, 0.315, 0.0015) * 0.82)
            img = over(img, (1.00, 0.70, 0.32), ellipse(cx, 0.332, 0.052, 0.058, 0.22) * 0.20)
            img = over(img, (0.73, 0.48, 0.18), rect(cx - 0.020, 0.286, cx + 0.020, 0.357, 0.010) * 0.88)
            img = over(img, (0.045, 0.030, 0.018), rect(cx + 0.020, 0.292, cx + 0.038, 0.360, 0.006) * 0.78)

        img = over(img, (0.06, 0.08, 0.12), line_y(horizon + 0.020, 0.0, 1.0, 0.0020) * 0.82)
        out = img.permute(2, 0, 1).reshape(3, h * w).contiguous()
        self._base_cache[cache_key] = out
        return out

    @torch.inference_mode()
    def frame_jpeg(self, t: float, quality: int = 82) -> bytes:
        start = time.perf_counter()
        w, h = self.width, self.height
        bg_img = self._base_scene(w, h, t)
        color_accum = torch.zeros((3, h * w), dtype=torch.float32, device=self.device)
        alpha_accum = torch.zeros((1, h * w), dtype=torch.float32, device=self.device)

        yaw = math.sin(t * 0.055) * 0.018
        c = math.cos(yaw)
        s = math.sin(yaw)
        x = self.xyz[:, 0] * c - (self.xyz[:, 2] - 3.1) * s
        z = self.xyz[:, 0] * s + (self.xyz[:, 2] - 3.1) * c + 3.1
        y = self.xyz[:, 1] + math.sin(t * 0.05) * 0.018
        inv_z = 1.0 / torch.clamp(z, min=0.7)
        px = ((x * inv_z) * 0.92 + 0.5) * w
        py = (0.55 - (y * inv_z) * 0.78) * h

        slow = 0.55 + 0.45 * torch.sin(t * self.speed + self.phase)
        fast = 0.55 + 0.45 * torch.sin(t * (self.speed * 5.7 + 0.8) + self.phase * 1.73)
        spike = torch.pow(torch.clamp(torch.sin(t * (self.speed * 1.8 + 0.2) + self.phase * 2.3), min=0.0), self.spike)
        alpha = torch.clamp(self.alpha_base * (0.72 + slow * 0.22 + fast * 0.06 + spike * 1.25), 0.01, 1.75)
        scale = torch.clamp(self.size * (6.8 * inv_z), 0.18, 10.5)

        base_x = px.long()
        base_y = py.long()
        ox = self.offsets[:, 0]
        oy = self.offsets[:, 1]
        sx = base_x[:, None] + ox[None, :]
        sy = base_y[:, None] + oy[None, :]
        inside = (sx >= 0) & (sx < w) & (sy >= 0) & (sy < h)
        flat = (sy * w + sx).clamp(0, h * w - 1)
        near_bias = torch.clamp(5.8 * inv_z, 0.38, 2.25)
        opacity = self.weights[None, :] * alpha[:, None] * scale[:, None] * near_bias[:, None] * 0.0048
        opacity = opacity * inside.float()
        contrib = self.color[:, :, None] * opacity[:, None, :]
        color_accum.scatter_add_(1, flat.reshape(1, -1).expand(3, -1), contrib.permute(1, 0, 2).reshape(3, -1))
        alpha_accum.scatter_add_(1, flat.reshape(1, -1), opacity.reshape(1, -1))

        alpha_img = torch.clamp(alpha_accum, 0.0, 0.96)
        src = color_accum / torch.clamp(alpha_accum, min=0.001)
        img = bg_img * (1.0 - alpha_img) + src * alpha_img
        img = torch.clamp(img * 0.88, 0.0, 1.0)
        img = torch.pow(img, 1.0 / 2.2)
        cpu = (img.reshape(3, h, w).permute(1, 2, 0) * 255).byte().cpu().numpy()
        render_done = time.perf_counter()
        buf = io.BytesIO()
        Image.fromarray(cpu, "RGB").save(buf, format="JPEG", quality=quality, optimize=False)
        self.last_ms = (render_done - start) * 1000.0
        self.last_encode_ms = (time.perf_counter() - render_done) * 1000.0
        self.frames += 1
        return buf.getvalue()

    def metrics(self) -> dict:
        return {
            "ok": True,
            "scene": "weaver_penthouse_3d_gaussian_splat_v3",
            "device": torch.cuda.get_device_name(0),
            "cuda": torch.version.cuda,
            "width": self.width,
            "height": self.height,
            "points": self.points,
            "components": self.part_counts,
            "frames": self.frames,
            "uptime_s": round(time.time() - self.started, 2),
            "last_render_ms": round(self.last_ms, 2),
            "last_encode_ms": round(self.last_encode_ms, 2),
            "target_fps": DEFAULT_FPS,
        }


renderer = CudaSplatRenderer()


def gpu_summary() -> str:
    try:
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=2,
        ).strip()
    except Exception as exc:
        return f"nvidia-smi unavailable: {exc}"


class Handler(BaseHTTPRequestHandler):
    server_version = "WeaverCudaSplat/1.0"

    def _json(self, status: int, data: dict) -> None:
        payload = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        if parsed.path in ("/", "/health", "/metrics"):
            data = renderer.metrics()
            data["gpu"] = gpu_summary()
            self._json(200, data)
            return
        if parsed.path == "/frame.jpg":
            self._resize_from_qs(qs)
            jpg = renderer.frame_jpeg(time.time())
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(jpg)))
            self.end_headers()
            self.wfile.write(jpg)
            return
        if parsed.path == "/stream.mjpg":
            self._resize_from_qs(qs)
            fps = max(1, min(int(qs.get("fps", [DEFAULT_FPS])[0]), 60))
            delay = 1.0 / fps
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=weaverframe")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            while True:
                frame_start = time.perf_counter()
                try:
                    jpg = renderer.frame_jpeg(time.time())
                    self.wfile.write(b"--weaverframe\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode("ascii"))
                    self.wfile.write(jpg)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return
                elapsed = time.perf_counter() - frame_start
                if elapsed < delay:
                    time.sleep(delay - elapsed)
            return
        self._json(404, {"ok": False, "error": "not found"})

    def _resize_from_qs(self, qs: dict) -> None:
        width = int(qs.get("w", [renderer.width])[0])
        height = int(qs.get("h", [renderer.height])[0])
        if width != renderer.width or height != renderer.height:
            renderer.resize(width, height)

    def log_message(self, fmt: str, *args) -> None:
        print(f"{self.address_string()} - {fmt % args}", flush=True)


if __name__ == "__main__":
    print(f"[weaver-render] CUDA renderer on {HOST}:{PORT} gpu={torch.cuda.get_device_name(0)}", flush=True)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
