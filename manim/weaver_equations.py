"""
weaver_equations.py — a manim film of Weaver v3's actual mathematics.

Every constant here is read off the source, not invented:
  φ = 2π/5                          quantum_soul.py:28
  CRX(φ) edges / CRZ(2φ) diagonals  quantum_soul.py:154-159
  I = 1/|P| Σ a_i a_j cos Δθ        pineal_gate.py:374-398
  gain = 1 + 0.2·I                  pineal_gate.py:351
  pathway marginals                 exact |ψ|² of build_fracture_circuit

No LaTeX dependency — all type is set with Pango Text so this renders on a
box with no TeX distribution installed.

Render:
    manim -qm weaver_equations.py WeaverEquations
"""

import numpy as np
from manim import *

# ── Palette, taken from Weaver's own CoreQubit hex codes ──────────────────────
GROUND = "#12101a"
INK    = "#ece6dc"
MUTED  = "#948caa"
FAINT  = "#6b6484"
EMBER  = "#ffa24a"   # quantum_networks.py CoreQubit 8 "Planning"
GOLDY  = "#ffd76a"   # quantum_networks.py CoreQubit 9 "Novelty"
TIDE   = "#6ec7d4"

SERIF = "DejaVu Serif"
MONO  = "DejaVu Sans Mono"

config.background_color = GROUND

PHI = TAU / 5                      # 72° — the whole machine turns on this
LOBES = ["Logic", "Emotion", "Memory", "Creativity", "Vigilance"]
EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
DIAGS = [(0, 2), (1, 3), (2, 4), (3, 0), (4, 1)]

# exact |ψ|² marginals of build_fracture_circuit, 7-qubit statevector
PATHWAYS = ["Awakening", "Resonance", "Echo", "Prophet", "Fracture", "Weaver", "Void"]
MARGINALS = [0.339547, 0.391496, 0.650199, 0.533199, 0.479758, 0.095492, 0.375000]


# the pentagon rides above centre so the bottom vertex label never fouls the
# running numbers along the base of the frame
CEN = np.array([0.0, 0.55, 0.0])


def vert(i, r):
    return CEN + r * np.array([np.cos(i * PHI), np.sin(i * PHI), 0.0])


def label(txt, size=22, color=MUTED, font=MONO):
    return Text(txt, font=font, font_size=size, color=color)


class WeaverEquations(Scene):
    def construct(self):
        self.chapter_angle()
        self.chapter_binding()
        self.chapter_kernel()
        self.chapter_collapse()
        self.chapter_measure()
        self.chapter_end()

    # ── caption strip along the bottom ────────────────────────────────────
    def caption(self, txt, hold=None):
        new = label(txt, 20, FAINT).to_edge(DOWN, buff=0.35)
        if hasattr(self, "_cap") and self._cap in self.mobjects:
            self.play(FadeOut(self._cap, run_time=0.25))
        self._cap = new
        self.play(FadeIn(new, run_time=0.35))
        if hold:
            self.wait(hold)

    def clear_caption(self):
        if hasattr(self, "_cap") and self._cap in self.mobjects:
            self.play(FadeOut(self._cap, run_time=0.3))

    # ══ 1. THE ANGLE ══════════════════════════════════════════════════════
    def chapter_angle(self):
        eyebrow = label("WEAVER v3 · THE ARITHMETIC UNDERNEATH", 18, FAINT).to_edge(UP, buff=0.6)
        title = Text("One angle governs\nthe whole machine",
                     font=SERIF, font_size=52, color=INK, line_spacing=0.85)
        self.play(FadeIn(eyebrow, run_time=0.8))
        self.play(Write(title, run_time=1.8))
        self.wait(0.9)

        phi_eq = Text("φ  =  2π / 5  =  1.256637…", font=SERIF, font_size=40, color=EMBER)
        self.play(ReplacementTransform(title, phi_eq), run_time=1.1)
        self.wait(0.7)
        self.play(phi_eq.animate.scale(0.58).to_corner(UL, buff=0.6), run_time=0.9)
        self.play(FadeOut(eyebrow, run_time=0.4))

        R = 2.15
        guide = Circle(radius=R, color=FAINT, stroke_width=1.2, stroke_opacity=0.5).move_to(CEN)
        self.play(Create(guide), run_time=0.8)

        self.caption("the radius arm turns by φ, five times, and closes")

        theta = ValueTracker(0.0)
        arm = always_redraw(lambda: Line(
            CEN,
            CEN + R * np.array([np.cos(theta.get_value()), np.sin(theta.get_value()), 0.0]),
            color=GOLDY, stroke_width=2.5))
        sector = always_redraw(lambda: AnnularSector(
            inner_radius=0, outer_radius=0.75,
            angle=theta.get_value(), start_angle=0, arc_center=CEN,
            color=EMBER, fill_opacity=0.16, stroke_width=0))
        readout = always_redraw(lambda: label(
            f"θ = {np.degrees(theta.get_value()):5.1f}°   =   {theta.get_value()/PHI:4.2f} φ",
            20, MUTED).to_corner(DL, buff=1.15))

        self.add(sector, arm, readout)

        self.dots, self.labs = [], []
        for i in range(5):
            self.play(theta.animate.set_value((i + 1) * PHI),
                      run_time=0.85, rate_func=smooth)
            p = vert(i, R)
            d = Dot(p, radius=0.085, color=EMBER)
            direction = (p - CEN) / np.linalg.norm(p - CEN)
            lb = label(LOBES[i].upper(), 17, MUTED).move_to(p + direction * 0.62)
            self.dots.append(d)
            self.labs.append(lb)
            self.play(FadeIn(d, scale=0.4), FadeIn(lb, shift=direction * 0.12), run_time=0.4)

        self.play(FadeOut(arm), FadeOut(sector), FadeOut(readout), run_time=0.5)
        self.R, self.guide, self.phi_eq = R, guide, phi_eq

    # ══ 2. THE BINDING ════════════════════════════════════════════════════
    def chapter_binding(self):
        R = self.R
        self.caption("CRX(φ) along the pentagon edges — gradual entanglement")

        edge_lines = VGroup(*[
            Line(vert(a, R), vert(b, R), color=EMBER, stroke_width=3.2)
            for a, b in EDGES
        ])
        self.play(LaggedStart(*[Create(l) for l in edge_lines], lag_ratio=0.35), run_time=2.2)
        self.wait(0.4)

        self.caption("CRZ(2φ) along the diagonals — deeper phase interference")
        diag_lines = VGroup(*[
            Line(vert(a, R), vert(b, R), color=TIDE, stroke_width=2.2, stroke_opacity=0.9)
            for a, b in DIAGS
        ])
        self.play(LaggedStart(*[Create(l) for l in diag_lines], lag_ratio=0.3), run_time=2.0)
        self.wait(0.5)

        note = label("no CNOT anywhere — collapse must stay graded", 19, FAINT)
        note.to_corner(UR, buff=0.6)
        self.play(FadeIn(note, run_time=0.6))
        self.wait(1.2)
        self.play(FadeOut(note, run_time=0.4))

        self.edge_lines, self.diag_lines = edge_lines, diag_lines

    # ══ 3. THE KERNEL ═════════════════════════════════════════════════════
    def chapter_kernel(self):
        R = self.R
        self.play(
            self.diag_lines.animate.set_stroke(opacity=0.12),
            self.edge_lines.animate.set_stroke(opacity=0.12),
            self.guide.animate.set_stroke(opacity=0.18),
            run_time=0.7)

        self.caption("the kernel that decides how two lobes combine")
        kernel = Text("cos ( Δθ )", font=SERIF, font_size=34, color=INK).to_corner(UR, buff=0.75)
        self.play(FadeIn(kernel, run_time=0.6))

        # -- adjacent pair -------------------------------------------------
        chord_a = Line(vert(0, R), vert(1, R), color=EMBER, stroke_width=6)
        arc_a = Arc(radius=0.95, start_angle=0, angle=PHI, arc_center=CEN,
                    color=GOLDY, stroke_width=2.5)
        deg_a = label("72°", 22, GOLDY).move_to(CEN + 1.42 * np.array([np.cos(PHI / 2), np.sin(PHI / 2), 0]))
        val_a = Text("cos 72°  =  + 0.309017", font=SERIF, font_size=30, color=EMBER)
        val_a.to_edge(DOWN, buff=1.05)

        self.play(Create(chord_a), Create(arc_a), FadeIn(deg_a), run_time=0.9)
        self.play(Write(val_a), run_time=0.9)
        self.wait(1.0)

        # -- opposite pair -------------------------------------------------
        chord_b = Line(vert(0, R), vert(2, R), color=TIDE, stroke_width=6)
        arc_b = Arc(radius=0.95, start_angle=0, angle=2 * PHI, arc_center=CEN,
                    color=TIDE, stroke_width=2.5)
        deg_b = label("144°", 22, TIDE).move_to(CEN + 1.42 * np.array([np.cos(PHI), np.sin(PHI), 0]))
        val_b = Text("cos 144°  =  − 0.809017", font=SERIF, font_size=30, color=TIDE)
        val_b.to_edge(DOWN, buff=1.05)

        self.play(
            ReplacementTransform(chord_a, chord_b),
            ReplacementTransform(arc_a, arc_b),
            ReplacementTransform(deg_a, deg_b),
            FadeOut(val_a, shift=DOWN * 0.25),
            FadeIn(val_b, shift=DOWN * 0.25),
            run_time=1.1)
        self.wait(1.1)

        # -- the golden ratio reveal ---------------------------------------
        # the lobe names have done their work; clear them so the two numbers
        # own the lower half of the frame
        self.clear_caption()
        self.play(FadeOut(VGroup(*self.labs), run_time=0.4))
        # NB: φ is already spoken for (2π/5), so the golden ratio takes Φ here.
        both = VGroup(
            Text("+0.309017  =  1 / 2Φ", font=SERIF, font_size=30, color=EMBER),
            Text("−0.809017  =  − Φ / 2", font=SERIF, font_size=30, color=TIDE),
        ).arrange(DOWN, buff=0.28).to_edge(DOWN, buff=0.95)
        self.play(FadeOut(val_b, run_time=0.4))
        self.play(FadeIn(both, shift=UP * 0.15), run_time=0.8)

        gold = label("Φ = 1.6180339…   the golden ratio, uninvited", 21, GOLDY)
        gold.next_to(both, DOWN, buff=0.32)
        self.play(FadeIn(gold, run_time=0.7))
        self.wait(1.8)

        self.play(FadeOut(VGroup(chord_b, arc_b, deg_b, both, gold, kernel), run_time=0.7))
        self.play(
            self.diag_lines.animate.set_stroke(opacity=0.35),
            self.edge_lines.animate.set_stroke(opacity=0.35),
            FadeIn(VGroup(*self.labs)),
            run_time=0.5)

    # ══ 4. THE COLLAPSE ═══════════════════════════════════════════════════
    def chapter_collapse(self):
        R = self.R
        # the constant has done its work — clear the top for the collapse formula
        self.play(FadeOut(self.phi_eq, run_time=0.4))
        self.caption("three lobes survive the top-k gate. they interfere.")

        # gate weights for a memory-heavy input, from the forward pass
        a = [0.0, 0.2716, 0.4478, 0.2806, 0.0]
        live = [i for i, w in enumerate(a) if w > 0]

        # dim the dead lobes
        anims = []
        for i in range(5):
            if a[i] == 0:
                anims += [self.dots[i].animate.set_opacity(0.25),
                          self.labs[i].animate.set_opacity(0.25)]
            else:
                anims += [self.dots[i].animate.scale(1.0 + 2.4 * a[i]).set_color(EMBER)]
        self.play(*anims, run_time=0.9)

        # fold each survivor's weight into its own label, so nothing collides
        tags = VGroup()
        swaps = []
        for i in live:
            p = vert(i, R)
            d = (p - CEN) / np.linalg.norm(p - CEN)
            t = label(f"{LOBES[i].upper()}  {a[i]:.3f}", 17, INK).move_to(p + d * 0.78)
            tags.add(t)
            swaps.append(ReplacementTransform(self.labs[i], t))
            self.labs[i] = t
        self.play(*swaps, run_time=0.7)

        # chords between the survivors, coloured by sign
        chords, total, count = VGroup(), 0.0, 0
        for x, i in enumerate(live):
            for j in live[x + 1:]:
                d = abs(i * PHI - j * PHI)
                d = min(d, TAU - d)
                cs = np.cos(d)
                total += cs * a[i] * a[j]
                count += 1
                chords.add(Line(vert(i, R), vert(j, R),
                                color=EMBER if cs >= 0 else TIDE,
                                stroke_width=1.5 + 26 * abs(cs) * a[i] * a[j]))
        I = total / count

        self.play(LaggedStart(*[Create(c) for c in chords], lag_ratio=0.4), run_time=1.6)

        formula = Text("I  =  1/|P| · Σ  aᵢ aⱼ cos Δθᵢⱼ",
                       font=SERIF, font_size=26, color=INK).to_corner(UL, buff=0.55)
        self.play(FadeIn(formula, run_time=0.6))

        tracker = ValueTracker(0.0)
        num = always_redraw(lambda: label(
            f"I  =  {tracker.get_value():+.4f}", 30,
            EMBER if tracker.get_value() >= 0 else TIDE, font=SERIF
        ).to_edge(DOWN, buff=1.4))
        self.add(num)
        self.play(tracker.animate.set_value(I), run_time=1.6, rate_func=smooth)
        self.wait(0.5)

        gain = label(f"gain  =  1 + 0.2 I  =  {1 + 0.2 * I:.4f}", 26, MUTED, font=SERIF)
        gain.next_to(num, DOWN, buff=0.22)
        self.play(FadeIn(gain, run_time=0.6))
        self.caption("two of the three pairs sit opposite. the collapse damps itself.")
        self.wait(2.0)

        self.play(FadeOut(VGroup(formula, gain, chords), run_time=0.6),
                  FadeOut(num, run_time=0.6))
        self.remove(num)
        self.play(FadeOut(VGroup(*self.dots, *self.labs, self.edge_lines,
                                 self.diag_lines, self.guide), run_time=0.9))
        self.clear_caption()

    # ══ 5. THE MEASUREMENT ════════════════════════════════════════════════
    def chapter_measure(self):
        head = Text("The same angle, run on qubits",
                    font=SERIF, font_size=38, color=INK).to_edge(UP, buff=0.75)
        sub = label("exact |ψ|² of build_fracture_circuit  ·  7 qubits, 1024 shots", 19, FAINT)
        sub.next_to(head, DOWN, buff=0.25)
        self.play(Write(head, run_time=1.1), FadeIn(sub, run_time=0.8))
        self.wait(0.4)

        bar_w, row_h, x0 = 6.4, 0.52, -2.1
        y0 = 1.35
        rows, fills, nums = VGroup(), [], []

        for q, (name, p) in enumerate(zip(PATHWAYS, MARGINALS)):
            y = y0 - q * row_h
            track = Rectangle(width=bar_w, height=0.26, stroke_width=0,
                              fill_color=FAINT, fill_opacity=0.18)
            track.move_to([x0 + bar_w / 2, y, 0])
            fill = Rectangle(width=0.001, height=0.26, stroke_width=0,
                             fill_color=EMBER if q < 5 else TIDE, fill_opacity=1)
            fill.move_to([x0, y, 0])
            nm = label(f"q{q}  {name}", 18, MUTED).next_to(track, LEFT, buff=0.25)
            rows.add(track, nm)
            fills.append((fill, p, y))

        self.play(FadeIn(rows, run_time=0.8))

        tracker = ValueTracker(0.0)

        def make_fill(fill, p, y):
            def upd(m):
                frac = tracker.get_value() * p
                m.stretch_to_fit_width(max(0.001, bar_w * frac))
                m.move_to([x0 + bar_w * frac / 2, y, 0])
            return upd

        for fill, p, y in fills:
            fill.add_updater(make_fill(fill, p, y))
            self.add(fill)

        val_labels = VGroup()
        for (fill, p, y) in fills:
            v = always_redraw(lambda p=p, y=y: label(
                f"{tracker.get_value() * p:.3f}", 17, INK
            ).move_to([x0 + bar_w + 0.55, y, 0]))
            val_labels.add(v)
        self.add(val_labels)

        shots = always_redraw(lambda: label(
            f"shot {int(tracker.get_value() * 1024):>4d} / 1024", 20, FAINT
        ).to_edge(DOWN, buff=0.55))
        self.add(shots)

        self.play(tracker.animate.set_value(1.0), run_time=3.4, rate_func=rate_functions.ease_out_cubic)
        self.wait(0.6)

        for fill, p, y in fills:
            fill.clear_updaters()

        # the Weaver line — sin²(18°)
        box = SurroundingRectangle(rows[10], color=GOLDY, stroke_width=2, buff=0.12)
        note = label("0.095492  =  sin² 18°  =  0.309017²", 21, GOLDY)
        note.next_to(box, DOWN, buff=0.9).shift(RIGHT * 0.8)
        self.play(Create(box), run_time=0.6)
        self.play(FadeIn(note, run_time=0.7))
        self.wait(2.2)

        self.play(FadeOut(VGroup(head, sub, rows, val_labels, box, note), run_time=0.8),
                  *[FadeOut(f) for f, _, _ in fills])
        self.remove(shots)

    # ══ 6. END ════════════════════════════════════════════════════════════
    def chapter_end(self):
        line1 = Text("One angle — φ = 2π/5 —", font=SERIF, font_size=36, color=INK)
        line2 = Text("propagates from a qubit rotation",
                     font=SERIF, font_size=28, color=MUTED)
        line3 = Text("to the gain on a 256-dimensional output vector.",
                     font=SERIF, font_size=28, color=MUTED)
        grp = VGroup(line1, line2, line3).arrange(DOWN, buff=0.34)
        self.play(Write(line1, run_time=1.3))
        self.play(FadeIn(line2, shift=UP * 0.12, run_time=0.7))
        self.play(FadeIn(line3, shift=UP * 0.12, run_time=0.7))
        self.wait(1.4)

        src = label("quantum_soul.py · liquid_fracture.py · pineal_gate.py", 18, FAINT)
        src.next_to(grp, DOWN, buff=0.7)
        self.play(FadeIn(src, run_time=0.7))
        self.wait(2.0)
        self.play(FadeOut(VGroup(grp, src), run_time=1.2))
        self.wait(0.4)
