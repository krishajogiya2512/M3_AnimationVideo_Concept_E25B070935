"""Scene 4 (146.50s - 221.80s): Worked Example f(x, y) = x^2 + y^2 at Point (1, 1)."""

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

# Polar mesh for crisp symmetric paraboloid f(x, y) = x^2 + y^2
_r = np.linspace(0.0, 2.0, 26)
_th = np.linspace(0.0, 2.0 * np.pi, 72)
_R, _TH = np.meshgrid(_r, _th, indexing="ij")
_BX = _R * np.cos(_TH)
_BY = _R * np.sin(_TH)
_BZ = 0.32 * (_BX ** 2 + _BY ** 2)


def _draw_paraboloid_2d_contours(canvas: Image.Image, box, t: float, alpha: float):
    """Draw 2D circular contour plot for f(x, y) = x^2 + y^2 with (1,1), (2,2), and unit vector u_hat."""
    if alpha <= 0.02:
        return
    x0, y0, x1, y1 = box
    cx, cy = x0 + 275, y0 + 635  # leaves generous room in Q1 for (3,3) vector tip below title
    scale = 148.0  # pixels per unit

    def to_px(mx, my):
        return cx + mx * scale, cy - my * scale

    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    a_i = int(round(255 * alpha))

    # Grid lines
    for g in range(-1, 4):
        gx_p, _ = to_px(g, 0)
        if x0 + 45 <= gx_p <= x1 - 45:
            d.line([(gx_p, y0 + 100), (gx_p, y1 - 50)], fill=(30, 44, 76, a_i), width=1)
    for g in range(-1, 4):
        _, gy_p = to_px(0, g)
        if y0 + 100 <= gy_p <= y1 - 50:
            d.line([(x0 + 45, gy_p), (x1 - 45, gy_p)], fill=(30, 44, 76, a_i), width=1)

    # Coordinate axes
    draw_arrow_2d(d, (x0 + 45, cy), (x1 - 45, cy), (*MUTED[:3], a_i), width=2, head_len=12)
    draw_arrow_2d(d, (cx, y1 - 50), (cx, y0 + 95), (*MUTED[:3], a_i), width=2, head_len=12)
    d.text((x1 - 34, cy), "x", font=get_font(22, "bold"), fill=(*TEXT[:3], a_i), anchor="mm")
    d.text((cx, y0 + 78), "y", font=get_font(22, "bold"), fill=(*TEXT[:3], a_i), anchor="mm")

    # Axis tick numbers 1, 2, 3
    for val in [1, 2, 3]:
        tx, _ = to_px(val, 0)
        if tx < x1 - 65:
            d.text((tx, cy + 20), str(val), font=get_font(17, "medium"), fill=(*MUTED[:3], a_i), anchor="mm")
        _, ty = to_px(0, val)
        if ty > y0 + 115:
            d.text((cx - 22, ty), str(val), font=get_font(17, "medium"), fill=(*MUTED[:3], a_i), anchor="mm")

    # Concentric circular level curves x^2 + y^2 = c
    for c_val in [1, 2, 4, 6, 8]:
        rad_m = math.sqrt(c_val)
        pts = []
        for k in range(181):
            ang = 2.0 * math.pi * k / 180.0
            px, py = to_px(rad_m * math.cos(ang), rad_m * math.sin(ang))
            if x0 + 45 <= px <= x1 - 45 and y0 + 100 <= py <= y1 - 50:
                pts.append((px, py))
            else:
                if len(pts) >= 2:
                    d.line(pts, fill=(*VIOLET[:3], int((230 if c_val == 2 else 155) * alpha)), width=3 if c_val == 2 else 2)
                pts = []
        if len(pts) >= 2:
            d.line(pts, fill=(*VIOLET[:3], int((230 if c_val == 2 else 155) * alpha)), width=3 if c_val == 2 else 2)

        # Label c value neatly in Q4/Q2
        lx, ly = to_px(rad_m * 0.707, -rad_m * 0.707) if c_val <= 4 else to_px(-rad_m * 0.55, rad_m * 0.83)
        if x0 + 75 <= lx <= x1 - 75 and y0 + 120 <= ly <= y1 - 65:
            d.text((lx, ly), f"f={c_val}", font=get_font(17, "bold"), fill=(*VIOLET[:3], a_i), anchor="mm")

    # Point P = (1, 1) on contour f = 2
    p11 = to_px(1.0, 1.0)
    # Tangent line at (1, 1): x + y = 2
    t_a = to_px(0.25, 1.75)
    t_b = to_px(1.75, 0.25)
    d.line([t_a, t_b], fill=(*AMBER[:3], int(185 * alpha)), width=3)

    # 90-degree square marker between tangent (-1/√2, 1/√2) and gradient (1/√2, 1/√2)
    u = 1.0 / math.sqrt(2.0)
    s_sq = 0.14
    sq1 = to_px(1.0 - s_sq * u, 1.0 + s_sq * u)
    sq2 = to_px(1.0 - s_sq * u + s_sq * u, 1.0 + s_sq * u + s_sq * u)
    sq3 = to_px(1.0 + s_sq * u, 1.0 + s_sq * u)
    d.line([sq1, sq2, sq3], fill=(*GOLD[:3], a_i), width=3)

    # 189.8 - 200.8: show test climb step along (2,2) vs along tangent (-1,1)
    a_climb = sub_progress(t, 192.0, 195.0) * (1.0 - sub_progress(t, 200.5, 201.5)) * alpha
    if a_climb > 0.02:
        # Moving bead climbing outward along (1,1) -> (1.7, 1.7) across f=2 -> f=5.78
        s_step = 0.65 * sub_progress(t, 192.5, 198.5, ease_in_out_cubic)
        bx, by = 1.0 + s_step * u, 1.0 + s_step * u
        b_px = to_px(bx, by)
        d.ellipse([b_px[0] - 8, b_px[1] - 8, b_px[0] + 8, b_px[1] + 8], fill=(*EMERALD[:3], int(255 * a_climb)))
        d.text((b_px[0] + 18, b_px[1] - 18), f"f = {bx*bx+by*by:.2f} ↑", font=get_font(18, "bold"), fill=(*EMERALD[:3], int(255 * a_climb)), anchor="lm")

    # Full vector (2, 2) from (1, 1) -> (3, 3) scaled gently or exact (1,1)->(2.5,2.5) with label (2,2)
    # Wait: (3,3) is at cx+534, cy-534 -> fits inside box! Let's check: cx=369, cy=677 -> cx+534=903 (<1040!), cy-534=143 (>102!). Exact (3,3) FITS!
    g_grow = sub_progress(t, 189.0, 191.2)
    # Shrink slightly in transparency after 206s when unit vector û is emphasized
    g_dim = 1.0 - 0.55 * sub_progress(t, 206.0, 208.5)
    g_tip = to_px(1.0 + 2.0 * g_grow, 1.0 + 2.0 * g_grow)
    draw_arrow_2d(d, p11, g_tip, (*CYAN[:3], int(255 * alpha * g_dim)), width=6, head_len=20)

    # Unit circle of radius 1 centered at (1, 1) + unit vector û = (1/√2, 1/√2)
    a_unit = sub_progress(t, 202.0, 205.5) * alpha
    if a_unit > 0.02:
        rad_u = scale * 1.0
        d.ellipse([p11[0] - rad_u, p11[1] - rad_u, p11[0] + rad_u, p11[1] + rad_u], outline=(*ROSE[:3], int(150 * a_unit)), width=2)
        u_len = lerp(2.0, 1.0, sub_progress(t, 209.5, 212.5, ease_in_out_cubic))
        u_tip = to_px(1.0 + u_len * u, 1.0 + u_len * u)
        draw_arrow_2d(d, p11, u_tip, (*ROSE[:3], int(255 * a_unit)), width=7, head_len=20)

    # Point (1,1) dot
    d.ellipse([p11[0] - 9, p11[1] - 9, p11[0] + 9, p11[1] + 9], fill=(*GOLD[:3], a_i), outline=(255, 255, 255, a_i), width=2)
    d.text((p11[0] - 46, p11[1] - 20), "(1, 1)", font=get_font(21, "bold"), fill=(*GOLD[:3], a_i), anchor="mm")

    out = Image.alpha_composite(canvas.convert("RGBA"), layer).convert("RGB")
    canvas.paste(out)

    draw_text_alpha(canvas, ((x0 + x1) // 2, y0 + 46), "Top-Down Contour Plot:  f(x, y) = x² + y²", get_font(25, "bold"), color=TEXT, alpha=alpha, anchor="mm")
    paste_tex(canvas, r"$\nabla f(1,1) = (2, 2)$", int(g_tip[0]) - 24, int(g_tip[1]) + 18, fontsize=19, color=CYAN, alpha=alpha * g_dim, anchor="cr")
    if a_unit > 0.15 and t >= 209.5:
        paste_tex(canvas, r"$\hat{u} = \left(\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}}\right)$", int(to_px(1.0 + u, 1.0 + u)[0]) + 22, int(to_px(1.0 + u, 1.0 + u)[1]) + 28, fontsize=18, color=ROSE, alpha=a_unit, anchor="cl")


def render_scene4(t: float) -> Image.Image:
    """Scene 4 (146.50 -> 221.80s): Worked Example f(x,y) = x^2+y^2, gradient at (1,1), unit direction."""
    canvas = new_canvas()
    fade_in = sub_progress(t, 146.50, 147.20)
    fade_out = 1.0 - sub_progress(t, 221.10, 221.80)
    master = fade_in * fade_out

    left_box = (64, 102, 1040, 1018)
    draw_rounded_card(canvas, left_box, radius=24, alpha=0.96 * master)

    # 146.5 - 189.2: 3D Paraboloid Bowl; 188.8 - 221.8: Top-Down 2D Contour Plot with exact vectors
    a_3d = (1.0 - sub_progress(t, 188.6, 190.2)) * master
    a_2d = sub_progress(t, 188.8, 190.4) * master

    if a_3d > 0.02:
        az = -32.0 + 12.0 * math.sin((t - 146.5) * 0.18)
        cam = Camera3D(azim_deg=az, elev_deg=26.0, scale=130.0, center=(535, 675), z_scale=1.18)
        d = ImageDraw.Draw(canvas)
        draw_3d_base_grid(d, cam, xlim=(-2.0, 2.0), ylim=(-2.0, 2.0), z0=0.0, step=1.0)
        draw_3d_axes(d, cam, xlim=(-2.2, 2.4), ylim=(-2.2, 2.4), zlim=(0.0, 2.2), labels=("x", "y", "f(x, y)"))

        items = []
        a_pt = sub_progress(t, 175.8, 177.5) * a_3d
        if a_pt > 0.02:
            items.append({"type": "point", "p": (1.0, 1.0, 0.0), "color": TEXT, "radius": 6, "label": "(1, 1)", "label_offset": (-42, 20), "alpha": a_pt})
            items.append({"type": "line", "pts": [(1.0, 1.0, 0.0), (1.0, 1.0, 0.32 * 2)], "color": GOLD, "width": 3, "dashed": True, "alpha": a_pt})
            items.append({"type": "point", "p": (1.0, 1.0, 0.32 * 2), "color": GOLD, "radius": 9, "label": "(1, 1, 2)", "label_offset": (12, -32), "alpha": a_pt})

        a_vec = sub_progress(t, 181.5, 184.5) * a_3d
        if a_vec > 0.02:
            items.append({"type": "arrow", "p0": (1.0, 1.0, 0.02), "p1": (1.0 + 0.85 * a_vec, 1.0 + 0.85 * a_vec, 0.02), "color": CYAN, "width": 6, "label": r"$\nabla f(1,1)=(2,2)$", "is_tex": True, "label_offset": (18, 34), "alpha": a_vec})

        render_surface_and_items(canvas, cam, _BX, _BY, _BZ, z_range=(0.0, 1.35), contour_levels="polar_rings", alpha=0.92 * a_3d, extra_items=items)
        draw_text_alpha(canvas, (552, 146), "3D Paraboloid Surface:  f(x, y) = x² + y²", get_font(25, "bold"), color=TEXT, alpha=a_3d, anchor="mm")

    if a_2d > 0.02:
        _draw_paraboloid_2d_contours(canvas, left_box, t, a_2d)

    # Right Step-by-Step Derivation Panel (1070..1856)
    draw_rounded_card(canvas, (1070, 102, 1856, 1018), radius=24, alpha=0.96 * master)
    draw_text_alpha(canvas, (1112, 144), "WORKED MATHEMATICAL EXAMPLE", get_font(20, "bold"), color=CYAN, alpha=master)
    draw_text_alpha(canvas, (1112, 184), "Finding & Normalizing the Gradient", get_font(34, "bold"), color=TEXT, alpha=master)

    # Step 1: Function
    a_s1 = sub_progress(t, 148.5, 150.2) * master
    if a_s1 > 0.02:
        draw_rounded_card(canvas, (1112, 240, 1814, 336), radius=16, fill=(19, 28, 52), outline=BORDER, alpha=a_s1)
        draw_text_alpha(canvas, (1142, 256), "STEP 1  •  FUNCTION", get_font(16, "bold"), color=MUTED, alpha=a_s1)
        paste_tex(canvas, r"$f(x, y) = x^2 + y^2$", 1463, 304, fontsize=21, color=TEXT, alpha=a_s1, anchor="center")

    # Step 2: Partial Derivatives & General Gradient
    a_s2x = sub_progress(t, 164.5, 166.2) * master
    a_s2y = sub_progress(t, 167.2, 169.0) * master
    a_s2g = sub_progress(t, 170.4, 172.2) * master
    if a_s2x > 0.02:
        draw_rounded_card(canvas, (1112, 354, 1814, 528), radius=16, fill=(19, 28, 52), outline=BORDER, alpha=a_s2x)
        draw_text_alpha(canvas, (1142, 370), "STEP 2  •  PARTIAL DERIVATIVES", get_font(16, "bold"), color=MUTED, alpha=a_s2x)
        paste_tex(canvas, r"$\frac{\partial f}{\partial x} = 2x$", 1285, 434, fontsize=19, color=AMBER, alpha=a_s2x, anchor="center")
        if a_s2y > 0.02:
            paste_tex(canvas, r"$\frac{\partial f}{\partial y} = 2y$", 1630, 434, fontsize=19, color=EMERALD, alpha=a_s2y, anchor="center")
        if a_s2g > 0.02:
            paste_tex(canvas, r"$\nabla f(x, y) = (2x, \; 2y)$", 1463, 496, fontsize=20, color=CYAN, alpha=a_s2g, anchor="center")

    # Step 3: Evaluate at (1, 1)
    a_s3 = sub_progress(t, 175.8, 178.0) * master
    a_s3_res = sub_progress(t, 179.5, 181.5) * master
    if a_s3 > 0.02:
        draw_rounded_card(canvas, (1112, 546, 1814, 676), radius=16, fill=(14, 34, 62), outline=CYAN, outline_width=2, alpha=a_s3)
        draw_text_alpha(canvas, (1142, 562), "STEP 3  •  EVALUATE AT POINT (1, 1)", get_font(16, "bold"), color=CYAN, alpha=a_s3)
        if a_s3_res < 0.05:
            paste_tex(canvas, r"$x = 1, \; y = 1$", 1463, 628, fontsize=21, color=GOLD, alpha=a_s3, anchor="center")
        else:
            paste_tex(canvas, r"$\nabla f(1, 1) = (2(1), \; 2(1)) = (2, \; 2)$", 1463, 628, fontsize=20, color=TEXT, alpha=a_s3_res, anchor="center")

    # Step 4: Magnitude
    a_s4 = sub_progress(t, 201.0, 203.2) * master
    a_s4_val = sub_progress(t, 206.4, 208.0) * master
    if a_s4 > 0.02:
        draw_rounded_card(canvas, (1112, 694, 1814, 818), radius=16, fill=(19, 28, 52), outline=BORDER, alpha=a_s4)
        draw_text_alpha(canvas, (1142, 710), "STEP 4  •  VECTOR MAGNITUDE", get_font(16, "bold"), color=MUTED, alpha=a_s4)
        tex_m = r"$\|\nabla f(1, 1)\| = \sqrt{2^2 + 2^2} = \sqrt{8} = 2\sqrt{2}$" if a_s4_val > 0.1 else r"$\|\nabla f(1, 1)\| = \sqrt{2^2 + 2^2}$"
        paste_tex(canvas, tex_m, 1463, 772, fontsize=17, color=GOLD, alpha=a_s4, anchor="center")

    # Step 5: Unit Direction Vector
    a_s5 = sub_progress(t, 209.8, 212.2) * master
    if a_s5 > 0.02:
        draw_rounded_card(canvas, (1112, 836, 1814, 984), radius=18, fill=(42, 22, 36), outline=ROSE, outline_width=2, alpha=a_s5)
        draw_text_alpha(canvas, (1142, 852), "STEP 5  •  UNIT DIRECTION OF STEEPEST ASCENT", get_font(16, "bold"), color=ROSE, alpha=a_s5)
        paste_tex(canvas, r"$\hat{u} = \frac{(2, 2)}{2\sqrt{2}} = \left( \frac{1}{\sqrt{2}}, \; \frac{1}{\sqrt{2}} \right) \approx (0.707, \; 0.707)$", 1463, 926, fontsize=16, color=TEXT, alpha=a_s5, anchor="center")

    draw_scene_chrome(canvas, t, TOTAL_DURATION, "SCENE 04 / 06", "Worked Example: f(x, y) = x² + y²")
    return canvas
