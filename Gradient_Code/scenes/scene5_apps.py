"""Scene 5 (221.80s - 256.30s): Applications in Machine Learning & Physics."""

import math
import numpy as np
from PIL import Image, ImageDraw
from styles import (
    CYAN, AMBER, EMERALD, VIOLET, ROSE, GOLD, TEXT, MUTED, DIM,
    BG_CARD, BORDER,
    get_font, new_canvas, draw_rounded_card, draw_text_alpha, paste_tex,
    draw_arrow_2d, draw_scene_chrome, sub_progress, ease_out_cubic, ease_in_out_cubic, clamp, lerp,
)
from surface3d import sample_palette
from timeline import TOTAL_DURATION


def _loss_fn(w1, w2):
    return 0.42 * w1 ** 2 + 0.85 * w2 ** 2 + 0.18 * w1 * w2


def _loss_grad(w1, w2):
    return 0.84 * w1 + 0.18 * w2, 1.70 * w2 + 0.18 * w1


# Precompute descent trajectory for ML card
_GD_PATH = []
_w1, _w2 = -1.75, 1.25
_lr = 0.24
for _ in range(18):
    _GD_PATH.append((_w1, _w2, _loss_fn(_w1, _w2)))
    g1, g2 = _loss_grad(_w1, _w2)
    _w1 -= _lr * g1
    _w2 -= _lr * g2


# Precompute crisp 2D Temperature Heatmap image for Physics card (560 x 350)
_HM_W, _HM_H = 740, 410
_hx = np.linspace(-2.0, 2.0, _HM_W)
_hy = np.linspace(-1.15, 1.15, _HM_H)
_HXX, _HYY = np.meshgrid(_hx, _hy)
# Hot source at (0.65, 0.20) + cooler secondary pocket at (-1.0, -0.3)
_TEMP = (
    88.0 * np.exp(-0.75 * ((_HXX - 0.65) ** 2 + (_HYY - 0.20) ** 2))
    + 36.0 * np.exp(-0.90 * ((_HXX + 0.95) ** 2 + (_HYY + 0.35) ** 2))
    + 18.0
)
_THERM_STOPS = [
    (18, 38, 98),     # Cool deep blue
    (28, 108, 178),   # Mild sky blue
    (52, 185, 168),   # Warm teal
    (245, 168, 36),   # Warm amber
    (239, 68, 68),    # Hot crimson
]
_t_norm = np.clip((_TEMP - 20.0) / 86.0, 0.0, 1.0)
_hm_rgb = np.clip(sample_palette(_t_norm, _THERM_STOPS), 0, 255).astype(np.uint8)
# Flip vertically so +y is up
_HM_IMG = Image.fromarray(_hm_rgb[::-1], mode="RGB")


def _temp_grad(x, y):
    e1 = 88.0 * np.exp(-0.75 * ((x - 0.65) ** 2 + (y - 0.20) ** 2))
    e2 = 36.0 * np.exp(-0.90 * ((x + 0.95) ** 2 + (y + 0.35) ** 2))
    gx = e1 * (-1.50 * (x - 0.65)) + e2 * (-1.80 * (x + 0.95))
    gy = e1 * (-1.50 * (y - 0.20)) + e2 * (-1.80 * (y + 0.35))
    return float(gx), float(gy)


def render_scene5(t: float) -> Image.Image:
    """Scene 5 (221.80 -> 256.30s): Real-World Applications in ML & Physics."""
    canvas = new_canvas()
    fade_in = sub_progress(t, 221.80, 222.50)
    fade_out = 1.0 - sub_progress(t, 255.60, 256.30)
    master = fade_in * fade_out

    # Scene Title Header
    draw_text_alpha(canvas, (960, 125), "WHY THE GRADIENT MATTERS IN PRACTICE", get_font(20, "bold"), color=CYAN, alpha=master, anchor="mm")
    draw_text_alpha(canvas, (960, 166), "From Artificial Intelligence to Physics", get_font(40, "bold"), color=TEXT, alpha=master, anchor="mm")

    # Left Card: Machine Learning — Gradient Descent (232.0+ focus)
    a_ml_card = sub_progress(t, 226.5, 228.5) * master
    ml_active = 0.65 + 0.35 * sub_progress(t, 232.2, 234.0)
    ml_box = (64, 215, 938, 1012)
    draw_rounded_card(canvas, ml_box, radius=24, fill=BG_CARD, outline=ROSE if t >= 232.2 else BORDER, outline_width=2 if t >= 232.2 else 1, alpha=0.96 * a_ml_card, accent_bar=ROSE)

    if a_ml_card > 0.02:
        draw_text_alpha(canvas, (110, 245), "1. AI & MACHINE LEARNING", get_font(20, "bold"), color=ROSE, alpha=a_ml_card * ml_active)
        draw_text_alpha(canvas, (110, 282), "Gradient Descent Minimizes Error", get_font(30, "bold"), color=TEXT, alpha=a_ml_card * ml_active)

        # Loss contour box inside ML card
        plot_box = (110, 345, 894, 775)
        draw_rounded_card(canvas, plot_box, radius=18, fill=(13, 19, 36), outline=BORDER, alpha=a_ml_card)
        cx, cy = 502, 560
        sc = 128.0

        def w_to_px(w1, w2):
            return cx + w1 * sc, cy - w2 * sc

        layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        a_i = int(round(255 * a_ml_card * ml_active))

        # Elliptic loss contours L(w1, w2) = c
        for c_val in [0.15, 0.45, 0.90, 1.50, 2.25]:
            pts = []
            for k in range(121):
                ang = 2.0 * math.pi * k / 120.0
                # Solve L(r cos, r sin) = c_val -> r = sqrt(c_val / L(cos, sin))
                denom = _loss_fn(math.cos(ang), math.sin(ang))
                r = math.sqrt(c_val / denom)
                px, py = w_to_px(r * math.cos(ang), r * math.sin(ang))
                if 125 <= px <= 880 and 360 <= py <= 760:
                    pts.append((px, py))
                else:
                    if len(pts) >= 2:
                        d.line(pts, fill=(*VIOLET[:3], int(150 * a_ml_card)), width=2)
                    pts = []
            if len(pts) >= 2:
                d.line(pts, fill=(*VIOLET[:3], int(150 * a_ml_card)), width=2)

        # Minimum star at (0, 0)
        d.ellipse([cx - 8, cy - 8, cx + 8, cy + 8], fill=(*EMERALD[:3], a_i), outline=(255, 255, 255, a_i), width=2)
        d.text((cx + 16, cy + 16), "Min Error", font=get_font(17, "bold"), fill=(*EMERALD[:3], a_i), anchor="lm")

        # Animate descent steps from 232.6 to 243.5
        prog = sub_progress(t, 232.6, 242.5, ease=lambda u: u)
        step_f = prog * (len(_GD_PATH) - 1)
        idx_int = int(math.floor(step_f))
        frac = step_f - idx_int

        trail = [w_to_px(_GD_PATH[k][0], _GD_PATH[k][1]) for k in range(idx_int + 1)]
        if idx_int < len(_GD_PATH) - 1:
            cw1 = lerp(_GD_PATH[idx_int][0], _GD_PATH[idx_int + 1][0], frac)
            cw2 = lerp(_GD_PATH[idx_int][1], _GD_PATH[idx_int + 1][1], frac)
            trail.append(w_to_px(cw1, cw2))
        else:
            cw1, cw2 = _GD_PATH[-1][0], _GD_PATH[-1][1]

        if len(trail) >= 2:
            d.line(trail, fill=(*GOLD[:3], a_i), width=4)
        for p_step in trail[:-1]:
            d.ellipse([p_step[0] - 4, p_step[1] - 4, p_step[0] + 4, p_step[1] + 4], fill=(*GOLD[:3], a_i))

        # Current weight ball + negative gradient arrow (-∇L)
        cur_px = trail[-1]
        g1, g2 = _loss_grad(cw1, cw2)
        gm = math.hypot(g1, g2)
        if gm > 0.08:
            arr_end = w_to_px(cw1 - 0.45 * g1 / gm, cw2 - 0.45 * g2 / gm)
            draw_arrow_2d(d, cur_px, arr_end, (*ROSE[:3], a_i), width=5, head_len=15)
        d.ellipse([cur_px[0] - 10, cur_px[1] - 10, cur_px[0] + 10, cur_px[1] + 10], fill=(*ROSE[:3], a_i), outline=(255, 255, 255, a_i), width=2)

        canvas = Image.alpha_composite(canvas.convert("RGBA"), layer).convert("RGB")
        cur_loss = _loss_fn(cw1, cw2)
        draw_text_alpha(canvas, (132, 368), f"Loss L(w) = {cur_loss:.3f}", get_font(20, "bold", mono=True), color=EMERALD if cur_loss < 0.1 else GOLD, alpha=a_ml_card)
        draw_text_alpha(canvas, (872, 368), "Descent direction: -grad L", get_font(18, "bold"), color=ROSE, alpha=a_ml_card, anchor="ra")

        # Update rule & explanation below plot
        draw_rounded_card(canvas, (110, 800, 894, 976), radius=16, fill=(20, 28, 52), outline=BORDER, alpha=a_ml_card)
        paste_tex(canvas, r"$w_{\mathrm{new}} = w - \eta\,\nabla L(w)$", 502, 855, fontsize=24, color=TEXT, alpha=a_ml_card, anchor="center")
        draw_text_alpha(canvas, (502, 932), "Opposite of gradient moves downhill to minimize error", get_font(20, "medium"), color=MUTED, alpha=a_ml_card, anchor="mm")

    # Right Card: Physics & Engineering — Temperature & Pressure Gradients (244.8+ focus)
    a_ph_card = sub_progress(t, 228.5, 230.5) * master
    ph_active = 0.65 + 0.35 * sub_progress(t, 244.8, 246.5)
    ph_box = (982, 215, 1856, 1012)
    draw_rounded_card(canvas, ph_box, radius=24, fill=BG_CARD, outline=AMBER if t >= 244.8 else BORDER, outline_width=2 if t >= 244.8 else 1, alpha=0.96 * a_ph_card, accent_bar=AMBER)

    if a_ph_card > 0.02:
        draw_text_alpha(canvas, (1028, 245), "2. PHYSICS & ENGINEERING", get_font(20, "bold"), color=AMBER, alpha=a_ph_card * ph_active)
        draw_text_alpha(canvas, (1028, 282), "Temperature & Pressure Fields", get_font(30, "bold"), color=TEXT, alpha=a_ph_card * ph_active)

        hm_x0, hm_y0 = 1048, 355
        draw_rounded_card(canvas, (hm_x0 - 12, hm_y0 - 10, hm_x0 + _HM_W + 12, hm_y0 + _HM_H + 10), radius=18, fill=(13, 19, 36), outline=BORDER, alpha=a_ph_card)
        # Paste precomputed thermal heatmap with opacity
        hm_rgba = _HM_IMG.convert("RGBA")
        if a_ph_card * ph_active < 0.99:
            r, g, b, a = hm_rgba.split()
            a = a.point(lambda v: int(v * a_ph_card * ph_active))
            hm_rgba = Image.merge("RGBA", (r, g, b, a))
        canvas.paste(hm_rgba, (hm_x0, hm_y0), hm_rgba)

        # Draw ∇T vector field over the heatmap pointing toward heat source
        layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        a_vec_ph = sub_progress(t, 245.0, 247.5) * a_ph_card * ph_active
        # Always show subtle grid of ∇T vectors (brighter once Physics section hits)
        vec_alpha = int(round((95 + 160 * sub_progress(t, 245.0, 247.2)) * a_ph_card))
        for gx_m in np.linspace(-1.6, 1.6, 8):
            for gy_m in np.linspace(-0.85, 0.85, 5):
                px = hm_x0 + ((gx_m + 2.0) / 4.0) * _HM_W
                py = hm_y0 + ((1.15 - gy_m) / 2.30) * _HM_H
                tx, ty = _temp_grad(gx_m, gy_m)
                tm = math.hypot(tx, ty)
                if tm > 8.0:
                    scale_l = min(38.0, 14.0 + tm * 0.25)
                    ux, uy = tx / tm, ty / tm
                    draw_arrow_2d(d, (px, py), (px + scale_l * ux, py - scale_l * uy), (*TEXT[:3], vec_alpha), width=3, head_len=9)

        # Label Hot & Cool regions cleanly between vector grid arrows
        hot_px = hm_x0 + ((0.65 + 2.0) / 4.0) * _HM_W
        hot_py = hm_y0 + ((1.15 - 0.20) / 2.30) * _HM_H
        d.rounded_rectangle([hot_px - 62, hot_py - 16, hot_px + 62, hot_py + 16], radius=8, fill=(14, 18, 34, int(180 * a_ph_card)))
        d.text((hot_px, hot_py), "HOT (T max)", font=get_font(18, "bold"), fill=(*TEXT[:3], int(255 * a_ph_card)), anchor="mm")

        cool_px = hm_x0 + ((-1.68 + 2.0) / 4.0) * _HM_W
        cool_py = hm_y0 + ((1.15 - 0.56) / 2.30) * _HM_H
        d.rounded_rectangle([cool_px - 36, cool_py - 14, cool_px + 36, cool_py + 14], radius=7, fill=(14, 18, 34, int(175 * a_ph_card)))
        d.text((cool_px, cool_py), "COOL", font=get_font(17, "bold"), fill=(*CYAN[:3], int(255 * a_ph_card)), anchor="mm")

        canvas = Image.alpha_composite(canvas.convert("RGBA"), layer).convert("RGB")

        draw_rounded_card(canvas, (1036, 800, 1800, 976), radius=16, fill=(20, 28, 52), outline=BORDER, alpha=a_ph_card)
        paste_tex(canvas, r"$\nabla T(x, y) = \left(\frac{\partial T}{\partial x}, \frac{\partial T}{\partial y}\right), \quad \vec{q} = -\kappa\nabla T$", 1418, 855, fontsize=21, color=TEXT, alpha=a_ph_card, anchor="center")
        draw_text_alpha(canvas, (1418, 932), "Temperature & pressure gradients point toward fastest rise", get_font(20, "medium"), color=MUTED, alpha=a_ph_card, anchor="mm")

    draw_scene_chrome(canvas, t, TOTAL_DURATION, "SCENE 05 / 06", "Applications: AI & Physics")
    return canvas
