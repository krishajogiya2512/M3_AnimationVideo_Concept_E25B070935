"""Scene 1 (8.80s - 42.10s): Introduction & Mountain Hook."""

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

# Stylized mountain terrain height function f(x, y)
def mountain_z(x, y):
    main_peak = 2.55 * np.exp(-0.34 * ((x - 0.35) ** 2 + (y - 0.30) ** 2))
    side_ridge = 1.05 * np.exp(-0.75 * ((x + 1.05) ** 2 + (y + 0.75) ** 2))
    return main_peak + side_ridge


def mountain_grad(x, y):
    e1 = 2.55 * np.exp(-0.34 * ((x - 0.35) ** 2 + (y - 0.30) ** 2))
    e2 = 1.05 * np.exp(-0.75 * ((x + 1.05) ** 2 + (y + 0.75) ** 2))
    gx = e1 * (-0.68 * (x - 0.35)) + e2 * (-1.50 * (x + 1.05))
    gy = e1 * (-0.68 * (y - 0.30)) + e2 * (-1.50 * (y + 0.75))
    return float(gx), float(gy)


_U = np.linspace(-2.2, 2.2, 52)
_MX, _MY = np.meshgrid(_U, _U)
_MZ = mountain_z(_MX, _MY)


def _trace_surface_path(x0, y0, ux, uy, length=0.95, n_pts=20):
    pts = []
    for k in range(n_pts):
        s = (k / (n_pts - 1)) * length
        px = x0 + ux * s
        py = y0 + uy * s
        pz = float(mountain_z(px, py)) + 0.05
        pts.append((px, py, pz))
    return pts


def render_scene1(t: float) -> Image.Image:
    """Scene 1 (8.80 -> 42.10s): Mountain intuition, paths, compass search, Gradient reveal."""
    canvas = new_canvas()
    fade_in = sub_progress(t, 8.80, 9.50)
    fade_out = 1.0 - sub_progress(t, 41.40, 42.10)
    master = fade_in * fade_out

    # Left 3D Mountain Card
    draw_rounded_card(canvas, (64, 102, 1130, 1018), radius=24, alpha=0.96 * master)

    # Gentle camera orbit
    az = -52.0 + 10.0 * math.sin((t - 8.80) * 0.14)
    cam = Camera3D(azim_deg=az, elev_deg=26.0, scale=142.0, center=(595, 675), z_scale=1.12)
    d = ImageDraw.Draw(canvas)
    draw_3d_base_grid(d, cam, xlim=(-2.2, 2.2), ylim=(-2.2, 2.2), z0=0.0, step=1.1)
    draw_3d_axes(d, cam, xlim=(-2.35, 2.55), ylim=(-2.35, 2.55), zlim=(0.0, 2.9), labels=("x", "y", "Height"))

    # Hiker position P mid-slope on the mountain so Uphill / Peak / Level stay well separated
    px0, py0 = -0.92, -0.52
    pz0 = float(mountain_z(px0, py0))
    gx, gy = mountain_grad(px0, py0)
    gmag = math.hypot(gx, gy)
    ux_best, uy_best = gx / gmag, gy / gmag
    # Orthogonal level direction (tangent to contour)
    ux_lvl, uy_lvl = -uy_best, ux_best

    items = [
        {"type": "line", "pts": [(px0, py0, 0.0), (px0, py0, pz0)], "color": GOLD, "width": 3, "dashed": True, "alpha": master},
        {"type": "point", "p": (0.35, 0.30, float(mountain_z(0.35, 0.30)) + 0.04), "color": GOLD, "radius": 8, "label": "Peak", "label_offset": (44, -18), "alpha": sub_progress(t, 10.5, 12.0) * master},
        {"type": "point", "p": (px0, py0, pz0 + 0.05), "color": TEXT, "radius": 9, "label": "You (P)", "label_offset": (-66, -18), "alpha": sub_progress(t, 9.4, 10.6) * master},
    ]

    # 15.50 - 23.20: Three candidate paths (uphill, downhill, barely changing)
    p_up = sub_progress(t, 16.5, 19.2)
    p_down = sub_progress(t, 19.2, 21.3)
    p_flat = sub_progress(t, 21.3, 23.2)
    p_dim_others = 1.0 - 0.65 * sub_progress(t, 28.2, 30.0)

    if p_up > 0.02:
        pts_up = _trace_surface_path(px0, py0, ux_best * 0.8 + ux_lvl * 0.6, uy_best * 0.8 + uy_lvl * 0.6, length=0.88 * p_up)
        items.append({"type": "line", "pts": pts_up, "color": EMERALD, "width": 5, "alpha": master * p_dim_others})
        if len(pts_up) >= 2:
            items.append({"type": "arrow", "p0": pts_up[-2], "p1": pts_up[-1], "color": EMERALD, "width": 5, "label": "Uphill" if (p_up > 0.7 and t < 28.5) else None, "label_offset": (12, -26), "alpha": master * p_dim_others})

    if p_down > 0.02:
        pts_dn = _trace_surface_path(px0, py0, -ux_best, -uy_best, length=0.85 * p_down)
        items.append({"type": "line", "pts": pts_dn, "color": ROSE, "width": 5, "alpha": master * p_dim_others})
        if len(pts_dn) >= 2:
            items.append({"type": "arrow", "p0": pts_dn[-2], "p1": pts_dn[-1], "color": ROSE, "width": 5, "label": "Downhill" if (p_down > 0.7 and t < 28.5) else None, "label_offset": (-48, 24), "alpha": master * p_dim_others})

    if p_flat > 0.02:
        # Follow true contour curve roughly perpendicular to gradient
        pts_fl = []
        cx_f, cy_f = px0, py0
        ds = (0.95 * p_flat) / 18.0
        for _ in range(19):
            pts_fl.append((cx_f, cy_f, float(mountain_z(cx_f, cy_f)) + 0.05))
            lgx, lgy = mountain_grad(cx_f, cy_f)
            lm = math.hypot(lgx, lgy) + 1e-6
            cx_f += (-lgy / lm) * ds
            cy_f += (lgx / lm) * ds
        items.append({"type": "line", "pts": pts_fl, "color": VIOLET, "width": 5, "alpha": master * p_dim_others})
        if len(pts_fl) >= 2:
            items.append({"type": "arrow", "p0": pts_fl[-2], "p1": pts_fl[-1], "color": VIOLET, "width": 5, "label": "Level" if (p_flat > 0.7 and t < 28.5) else None, "label_offset": (46, 18), "alpha": master * p_dim_others})

    # 23.20 - 28.06: Compass sweep searching for fastest climb; 28.06+: lock onto steepest ascent
    a_sweep = sub_progress(t, 23.2, 24.0) * (1.0 - sub_progress(t, 28.5, 29.5))
    best_ang = math.atan2(uy_best, ux_best)
    if t < 28.1:
        sweep_ang = best_ang + math.pi * 1.15 * math.sin((t - 23.2) * 2.1)
    else:
        sweep_ang = best_ang

    if a_sweep > 0.02:
        sx_dir, sy_dir = math.cos(sweep_ang), math.sin(sweep_ang)
        pts_sw = _trace_surface_path(px0, py0, sx_dir, sy_dir, length=0.92, n_pts=14)
        items.append({"type": "line", "pts": pts_sw, "color": GOLD, "width": 5, "alpha": a_sweep * master})
        items.append({"type": "arrow", "p0": pts_sw[-2], "p1": pts_sw[-1], "color": GOLD, "width": 5, "alpha": a_sweep * master})

    a_grad = sub_progress(t, 28.1, 29.8) * master
    if a_grad > 0.02:
        pts_gr = _trace_surface_path(px0, py0, ux_best, uy_best, length=1.05 * a_grad, n_pts=18)
        items.append({"type": "line", "pts": pts_gr, "color": CYAN, "width": 7, "alpha": master})
        if len(pts_gr) >= 2:
            items.append({"type": "arrow", "p0": pts_gr[-2], "p1": pts_gr[-1], "color": CYAN, "width": 7, "label": r"$\nabla f$ (Fastest)", "is_tex": True, "label_offset": (-26, -34), "alpha": master})

    render_surface_and_items(
        canvas, cam, _MX, _MY, _MZ,
        z_range=(0.0, 2.7),
        contour_levels=[0.4, 0.8, 1.2, 1.6, 2.0, 2.4],
        alpha=0.95 * master,
        extra_items=items,
    )

    # Right Explanatory Panel (1160..1856)
    draw_rounded_card(canvas, (1160, 102, 1856, 1018), radius=24, alpha=0.96 * master)

    draw_text_alpha(canvas, (1205, 148), "THE OPTIMIZATION QUESTION", get_font(20, "bold"), color=CYAN, alpha=master)
    draw_text_alpha(canvas, (1205, 190), "Which way climbs fastest?", get_font(38, "bold"), color=TEXT, alpha=master)

    # Path cards (appear as she mentions uphill / downhill / barely change height)
    path_cards = [
        (16.0, "Uphill Paths", "Height increases (Δz > 0)", EMERALD),
        (19.0, "Downhill Paths", "Height decreases (Δz < 0)", ROSE),
        (21.2, "Contour Paths", "Height stays nearly flat (Δz ≈ 0)", VIOLET),
    ]
    for idx, (t_show, title, desc, col) in enumerate(path_cards):
        ac = sub_progress(t, t_show, t_show + 0.8) * master
        if ac > 0.02:
            y0 = 265 + idx * 95
            draw_rounded_card(canvas, (1205, y0, 1810, y0 + 78), radius=14, fill=(21, 30, 56), outline=col, outline_width=2, alpha=ac * (0.55 + 0.45 * p_dim_others))
            draw_text_alpha(canvas, (1232, y0 + 18), title, get_font(22, "bold"), color=col, alpha=ac)
            draw_text_alpha(canvas, (1232, y0 + 46), desc, get_font(19, "regular"), color=MUTED, alpha=ac)

    # Live Slope Rate Meter (23.2 - 34.0)
    a_meter = sub_progress(t, 23.2, 24.2) * master
    if a_meter > 0.02:
        draw_rounded_card(canvas, (1205, 565, 1810, 695), radius=16, fill=(19, 28, 52), outline=BORDER, alpha=a_meter)
        cur_slope = gx * math.cos(sweep_ang) + gy * math.sin(sweep_ang)
        norm_rate = cur_slope / (gmag + 1e-9)  # in [-1, 1]
        draw_text_alpha(canvas, (1232, 588), "Slope Rate:", get_font(19, "demibold"), color=MUTED, alpha=a_meter)
        d = ImageDraw.Draw(canvas)
        bar_x0, bar_x1, bar_y = 1232, 1780, 646
        mid_x = (bar_x0 + bar_x1) // 2
        d.rounded_rectangle([bar_x0, bar_y - 12, bar_x1, bar_y + 12], radius=6, fill=(30, 42, 74))
        d.line([(mid_x, bar_y - 18), (mid_x, bar_y + 18)], fill=MUTED, width=2)
        fill_x = int(mid_x + norm_rate * ((bar_x1 - bar_x0) * 0.46))
        bar_col = CYAN if t >= 28.1 else (EMERALD if norm_rate >= 0 else ROSE)
        if fill_x >= mid_x:
            d.rounded_rectangle([mid_x, bar_y - 10, fill_x, bar_y + 10], radius=5, fill=bar_col)
        else:
            d.rounded_rectangle([fill_x, bar_y - 10, mid_x, bar_y + 10], radius=5, fill=bar_col)
        rate_lbl = "MAXIMUM (+100%)" if t >= 28.1 else f"{norm_rate * 100:+.0f}% of max slope"
        draw_text_alpha(canvas, (1780, 588), rate_lbl, get_font(20, "bold"), color=bar_col, alpha=a_meter, anchor="ra")

    # Gradient answer reveal (28.1+)
    a_ans = sub_progress(t, 28.5, 30.2) * master
    if a_ans > 0.02:
        draw_rounded_card(canvas, (1205, 720, 1810, 845), radius=18, fill=(14, 34, 62), outline=CYAN, outline_width=2, alpha=a_ans)
        draw_text_alpha(canvas, (1232, 742), "ANSWER: THE GRADIENT VECTOR", get_font(21, "bold"), color=CYAN, alpha=a_ans)
        paste_tex(canvas, r"$\nabla f$  points in the direction of steepest ascent", 1508, 804, fontsize=15, color=TEXT, alpha=a_ans, anchor="center")

    # Roadmap (35.0+)
    a_road = sub_progress(t, 35.0, 36.8) * master
    if a_road > 0.02:
        pills = ["1. What It Is", "2. How It Works", "3. Why Steepest"]
        for k, ptxt in enumerate(pills):
            ap = sub_progress(t, 35.2 + k * 1.1, 36.2 + k * 1.1) * master
            px = 1205 + k * 206
            draw_rounded_card(canvas, (px, 875, px + 192, 975), radius=14, fill=(22, 33, 62), outline=GOLD if k == 2 else BORDER, alpha=ap)
            draw_text_alpha(canvas, (px + 96, 925), ptxt, get_font(19, "bold"), color=TEXT, alpha=ap, anchor="mm")

    draw_scene_chrome(canvas, t, TOTAL_DURATION, "SCENE 01 / 06", "Hook: The Mountain Problem")
    return canvas
