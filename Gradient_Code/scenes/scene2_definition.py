"""Scene 2 (42.10s - 96.80s): Understanding the Gradient & Partial Derivatives."""

import math
import numpy as np
from PIL import Image, ImageDraw
from styles import (
    CYAN, AMBER, EMERALD, VIOLET, ROSE, GOLD, TEXT, MUTED, DIM,
    BG_CARD, BORDER,
    get_font, new_canvas, draw_rounded_card, draw_text_alpha, paste_tex,
    draw_arrow_2d, draw_scene_chrome, sub_progress, ease_out_cubic, ease_in_out_cubic, clamp, lerp,
)
from surface3d import Camera3D, draw_3d_base_grid, draw_3d_axes, render_surface_and_items
from timeline import TOTAL_DURATION


def surf_f(x, y):
    return 1.55 + 0.42 * np.sin(0.85 * x) * np.cos(0.85 * y) + 0.24 * x + 0.20 * y


def surf_grad(x, y):
    gx = 0.42 * 0.85 * np.cos(0.85 * x) * np.cos(0.85 * y) + 0.24
    gy = -0.42 * 0.85 * np.sin(0.85 * x) * np.sin(0.85 * y) + 0.20
    return float(gx), float(gy)


_U = np.linspace(-2.0, 2.0, 48)
_SX, _SY = np.meshgrid(_U, _U)
_SZ = surf_f(_SX, _SY)


def _draw_1d_curve_panel(canvas: Image.Image, box, alpha: float):
    """Draw a clean 1D function y = g(x) curve panel to contrast with 2D->3D surface."""
    if alpha <= 0.02:
        return
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2 + 40
    d = ImageDraw.Draw(canvas)
    # Axes
    draw_arrow_2d(d, (x0 + 90, cy + 160), (x1 - 90, cy + 160), MUTED, width=3, head_len=14)
    draw_arrow_2d(d, (cx, cy + 210), (cx, y0 + 120), MUTED, width=3, head_len=14)
    draw_text_alpha(canvas, (x1 - 75, cy + 160), "x", get_font(26, "bold"), color=TEXT, alpha=alpha, anchor="mm")
    draw_text_alpha(canvas, (cx, y0 + 95), "y = g(x)", get_font(26, "bold"), color=TEXT, alpha=alpha, anchor="mm")

    xs = np.linspace(-3.0, 3.0, 140)
    ys = 0.22 * xs ** 2 + 0.35 * np.sin(1.3 * xs) + 0.5
    pts = [(cx + xv * 115, cy + 160 - yv * 95) for xv, yv in zip(xs, ys)]
    d.line(pts, fill=CYAN, width=5)

    # Tangent at x = 1.1
    xt = 1.1
    yt = 0.22 * xt ** 2 + 0.35 * math.sin(1.3 * xt) + 0.5
    slope = 0.44 * xt + 0.35 * 1.3 * math.cos(1.3 * xt)
    px, py = cx + xt * 115, cy + 160 - yt * 95
    t0 = (cx + (xt - 0.85) * 115, cy + 160 - (yt - 0.85 * slope) * 95)
    t1 = (cx + (xt + 0.85) * 115, cy + 160 - (yt + 0.85 * slope) * 95)
    d.line([t0, t1], fill=AMBER, width=4)
    d.ellipse([px - 8, py - 8, px + 8, py + 8], fill=GOLD, outline=TEXT, width=2)
    draw_text_alpha(canvas, (px + 20, py - 24), "1 input (x) -> 1 slope g'(x)", get_font(22, "demibold"), color=AMBER, alpha=alpha)


def render_scene2(t: float) -> Image.Image:
    """Scene 2 (42.10 -> 96.80s): Function of two variables -> 3D surface -> Partial derivatives -> ∇f."""
    canvas = new_canvas()
    fade_in = sub_progress(t, 42.10, 42.80)
    fade_out = 1.0 - sub_progress(t, 96.10, 96.80)
    master = fade_in * fade_out

    # Left visual panel
    left_box = (64, 102, 1070, 1018)
    draw_rounded_card(canvas, left_box, radius=24, alpha=0.96 * master)

    # 45.7 - 52.8: Contrast 1D curve first, then transition cleanly to 3D surface around 52.5s
    a_1d = sub_progress(t, 45.5, 47.0) * (1.0 - sub_progress(t, 52.2, 53.8)) * master
    a_3d = sub_progress(t, 52.4, 54.4) * master

    if a_1d > 0.02:
        draw_text_alpha(canvas, (567, 152), "Single-Variable Function: 1D Curve", get_font(28, "bold"), color=CYAN, alpha=a_1d, anchor="mm")
        _draw_1d_curve_panel(canvas, left_box, a_1d)

    if a_3d > 0.02:
        az = -46.0 + 6.0 * math.sin((t - 52.4) * 0.22)
        cam = Camera3D(azim_deg=az, elev_deg=26.0, scale=142.0, center=(555, 680), z_scale=1.15)
        d = ImageDraw.Draw(canvas)
        draw_3d_base_grid(d, cam, xlim=(-2.0, 2.0), ylim=(-2.0, 2.0), z0=0.0, step=1.0)
        draw_3d_axes(d, cam, xlim=(-2.2, 2.5), ylim=(-2.2, 2.5), zlim=(0.0, 2.8), labels=("x", "y", "f(x, y)"))

        x0, y0 = -0.15, -0.15
        z0 = float(surf_f(x0, y0))
        gx, gy = surf_grad(x0, y0)

        items = []
        # 57.8+: position (x,y) on floor + height f(x,y) plumb line
        a_pos = sub_progress(t, 57.8, 59.5) * a_3d
        if a_pos > 0.02:
            items.append({"type": "point", "p": (x0, y0, 0.0), "color": TEXT, "radius": 7, "label": "(x, y)", "label_offset": (-44, 22), "alpha": a_pos})
            items.append({"type": "line", "pts": [(x0, y0, 0.0), (x0, y0, z0)], "color": GOLD, "width": 3, "dashed": True, "alpha": a_pos})
            items.append({"type": "point", "p": (x0, y0, z0 + 0.03), "color": GOLD, "radius": 9, "label": "z = f(x, y)", "label_offset": (-66, -28), "alpha": a_pos})

        # 64.6+: x-slice (Amber) and y-slice (Emerald) cross sections
        a_slice_x = sub_progress(t, 64.8, 67.0) * a_3d
        a_slice_y = sub_progress(t, 66.2, 68.5) * a_3d
        if a_slice_x > 0.02:
            xs_c = np.linspace(-1.95, 1.95, 45)
            pts_xc = [(float(xv), y0, float(surf_f(xv, y0)) + 0.04) for xv in xs_c]
            items.append({"type": "line", "pts": pts_xc, "color": AMBER, "width": 5, "alpha": a_slice_x})
        if a_slice_y > 0.02:
            ys_c = np.linspace(-1.95, 1.95, 45)
            pts_yc = [(x0, float(yv), float(surf_f(x0, yv)) + 0.04) for yv in ys_c]
            items.append({"type": "line", "pts": pts_yc, "color": EMERALD, "width": 5, "alpha": a_slice_y})

        # 83.3+: show ∂f/∂x component arrow along x on surface & floor
        a_vx = sub_progress(t, 83.3, 85.0) * a_3d
        a_vy = sub_progress(t, 88.1, 89.6) * a_3d
        a_sum = sub_progress(t, 91.6, 93.4) * a_3d

        scale_v = 2.0
        dx_len = gx * scale_v
        dy_len = gy * scale_v
        if a_vx > 0.02:
            # Tangent arrow on surface along x
            items.append({"type": "arrow", "p0": (x0, y0, z0 + 0.04), "p1": (x0 + 0.75 * a_vx, y0, z0 + 0.04 + gx * 0.75 * a_vx), "color": AMBER, "width": 5, "alpha": a_vx})
            # Floor projection vector
            items.append({"type": "arrow", "p0": (x0, y0, 0.02), "p1": (x0 + dx_len * a_vx, y0, 0.02), "color": AMBER, "width": 5, "label": r"$\frac{\partial f}{\partial x}$", "is_tex": True, "label_offset": (38, 28), "alpha": a_vx})

        if a_vy > 0.02:
            items.append({"type": "arrow", "p0": (x0, y0, z0 + 0.04), "p1": (x0, y0 + 0.75 * a_vy, z0 + 0.04 + gy * 0.75 * a_vy), "color": EMERALD, "width": 5, "alpha": a_vy})
            items.append({"type": "arrow", "p0": (x0, y0, 0.02), "p1": (x0, y0 + dy_len * a_vy, 0.02), "color": EMERALD, "width": 5, "label": r"$\frac{\partial f}{\partial y}$", "is_tex": True, "label_offset": (38, -28), "alpha": a_vy})

        if a_sum > 0.02:
            # Parallelogram dashed guides + diagonal resultant ∇f on floor
            items.append({"type": "line", "pts": [(x0 + dx_len, y0, 0.02), (x0 + dx_len, y0 + dy_len, 0.02)], "color": DIM, "width": 2, "dashed": True, "alpha": a_sum})
            items.append({"type": "line", "pts": [(x0, y0 + dy_len, 0.02), (x0 + dx_len, y0 + dy_len, 0.02)], "color": DIM, "width": 2, "dashed": True, "alpha": a_sum})
            items.append({"type": "arrow", "p0": (x0, y0, 0.03), "p1": (x0 + dx_len * a_sum, y0 + dy_len * a_sum, 0.03), "color": CYAN, "width": 7, "label": r"$\nabla f$", "is_tex": True, "label_offset": (34, -22), "alpha": a_sum})

        render_surface_and_items(
            canvas, cam, _SX, _SY, _SZ,
            z_range=(0.4, 2.7),
            contour_levels=[1.0, 1.4, 1.8, 2.2],
            alpha=0.88 * a_3d,
            extra_items=items,
        )
        draw_text_alpha(canvas, (567, 146), "Two-Variable Function: 3D Surface z = f(x, y)", get_font(26, "bold"), color=TEXT, alpha=a_3d, anchor="mm")

    # Right Formal Definition Panel (1100..1856)
    draw_rounded_card(canvas, (1100, 102, 1856, 1018), radius=24, alpha=0.96 * master)
    draw_text_alpha(canvas, (1145, 148), "FORMAL DEFINITION", get_font(20, "bold"), color=CYAN, alpha=master)
    draw_text_alpha(canvas, (1145, 190), "What is the Gradient?", get_font(36, "bold"), color=TEXT, alpha=master)

    # Step 1: Two-variable notation (45.7+)
    a_fxy = sub_progress(t, 45.7, 47.2) * master
    if a_fxy > 0.02:
        draw_rounded_card(canvas, (1145, 255, 1810, 365), radius=16, fill=(20, 30, 56), outline=BORDER, alpha=a_fxy)
        paste_tex(canvas, r"$z = f(x, y)$", 1280, 310, fontsize=26, color=GOLD, alpha=a_fxy, anchor="center")
        draw_text_alpha(canvas, (1430, 285), "(x, y) : Position on plane", get_font(20, "medium"), color=MUTED, alpha=a_fxy)
        draw_text_alpha(canvas, (1430, 318), "f(x, y) : Surface height z", get_font(20, "medium"), color=TEXT, alpha=a_fxy)

    # Step 2: Gradient vector formula (70.1+)
    a_form = sub_progress(t, 70.1, 72.2) * master
    if a_form > 0.02:
        draw_rounded_card(canvas, (1145, 392, 1810, 560), radius=18, fill=(14, 34, 62), outline=CYAN, outline_width=2, alpha=a_form)
        draw_text_alpha(canvas, (1175, 415), "GRADIENT VECTOR", get_font(18, "bold"), color=CYAN, alpha=a_form)
        paste_tex(canvas, r"$\nabla f(x, y) = \left( \frac{\partial f}{\partial x}, \; \frac{\partial f}{\partial y} \right)$", 1478, 492, fontsize=30, color=TEXT, alpha=a_form, anchor="center")

    # Step 3: Component breakdown cards (83.3+ and 88.1+)
    a_c1 = sub_progress(t, 83.3, 84.8) * master
    if a_c1 > 0.02:
        draw_rounded_card(canvas, (1145, 585, 1810, 695), radius=16, fill=(26, 30, 50), outline=AMBER, outline_width=2, alpha=a_c1)
        paste_tex(canvas, r"$\frac{\partial f}{\partial x}$", 1210, 640, fontsize=25, color=AMBER, alpha=a_c1, anchor="center")
        draw_text_alpha(canvas, (1285, 614), "1st Component (x-direction)", get_font(22, "bold"), color=AMBER, alpha=a_c1)
        draw_text_alpha(canvas, (1285, 648), "Rate of change holding y constant", get_font(19, "regular"), color=MUTED, alpha=a_c1)

    a_c2 = sub_progress(t, 88.1, 89.6) * master
    if a_c2 > 0.02:
        draw_rounded_card(canvas, (1145, 718, 1810, 828), radius=16, fill=(18, 36, 52), outline=EMERALD, outline_width=2, alpha=a_c2)
        paste_tex(canvas, r"$\frac{\partial f}{\partial y}$", 1210, 773, fontsize=25, color=EMERALD, alpha=a_c2, anchor="center")
        draw_text_alpha(canvas, (1285, 747), "2nd Component (y-direction)", get_font(22, "bold"), color=EMERALD, alpha=a_c2)
        draw_text_alpha(canvas, (1285, 781), "Rate of change holding x constant", get_font(19, "regular"), color=MUTED, alpha=a_c2)

    # Step 4: Vector combination badge (91.6+)
    a_comb = sub_progress(t, 91.6, 93.2) * master
    if a_comb > 0.02:
        draw_rounded_card(canvas, (1145, 852, 1810, 978), radius=16, fill=(18, 28, 54), outline=CYAN, outline_width=2, alpha=a_comb)
        paste_tex(canvas, r"$\nabla f = \frac{\partial f}{\partial x}\,\hat{i} \;+\; \frac{\partial f}{\partial y}\,\hat{j}$", 1478, 898, fontsize=23, color=CYAN, alpha=a_comb, anchor="center")
        draw_text_alpha(canvas, (1478, 948), "Combined vector points in direction of steepest ascent", get_font(19, "medium"), color=TEXT, alpha=a_comb, anchor="mm")

    draw_scene_chrome(canvas, t, TOTAL_DURATION, "SCENE 02 / 06", "Concept & Partial Derivatives")
    return canvas
