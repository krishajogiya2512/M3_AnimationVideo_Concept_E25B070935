"""Scene 3 (96.80s - 146.50s): Visual Intuition, Directional Derivative, and Contours."""

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


# Smooth elliptic hillside centered at (1.5, 1.4) so slope at (0,0) points neatly northeast
def hill_f(x, y):
    return 2.75 * np.exp(-0.22 * ((x - 1.35) ** 2 + 1.15 * (y - 1.15) ** 2))


def hill_grad(x, y):
    val = float(hill_f(x, y))
    gx = val * (-0.44 * (x - 1.35))
    gy = val * (-0.44 * 1.15 * (y - 1.15))
    return gx, gy


_U = np.linspace(-2.0, 2.2, 48)
_HX, _HY = np.meshgrid(_U, _U)
_HZ = hill_f(_HX, _HY)


def _draw_2d_contour_panel(canvas: Image.Image, box, t: float, alpha: float):
    """Draw crisp top-down 2D contour plot with 90-degree orthogonality and sliding contour bead."""
    if alpha <= 0.02:
        return
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) // 2 - 20, (y0 + y1) // 2 + 25
    scale = 175.0  # pixels per math unit

    def to_px(mx, my):
        return cx + mx * scale, cy - my * scale

    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    a_i = int(round(255 * alpha))

    # Subtle 2D coordinate grid
    for g in [-2, -1, 0, 1, 2]:
        gx_p, _ = to_px(g, 0)
        _, gy_p = to_px(0, g)
        d.line([(gx_p, y0 + 95), (gx_p, y1 - 55)], fill=(30, 44, 76, a_i), width=1)
        d.line([(x0 + 55, gy_p), (x1 - 55, gy_p)], fill=(30, 44, 76, a_i), width=1)

    # Axes
    draw_arrow_2d(d, (x0 + 55, cy), (x1 - 50, cy), (*MUTED[:3], a_i), width=2, head_len=12)
    draw_arrow_2d(d, (cx, y1 - 55), (cx, y0 + 90), (*MUTED[:3], a_i), width=2, head_len=12)

    # Concentric elliptic contour curves centered at peak (0.9, 0.8) for crystal clarity
    pk_x, pk_y = 0.85, 0.75
    radii = [0.42, 0.82, 1.24, 1.68, 2.12]
    levels_lbl = ["f = 2.6", "f = 2.2", "f = 1.7", "f = 1.2", "f = 0.7"]
    b_ratio = 0.86

    for idx, r in enumerate(radii):
        pts = []
        for k in range(121):
            ang = 2.0 * math.pi * k / 120.0
            mx = pk_x + r * math.cos(ang)
            my = pk_y + r * b_ratio * math.sin(ang)
            if x0 + 55 <= to_px(mx, my)[0] <= x1 - 55 and y0 + 95 <= to_px(mx, my)[1] <= y1 - 55:
                pts.append(to_px(mx, my))
            else:
                if len(pts) >= 2:
                    d.line(pts, fill=(*VIOLET[:3], int(190 * alpha)), width=3)
                pts = []
        if len(pts) >= 2:
            d.line(pts, fill=(*VIOLET[:3], int(190 * alpha)), width=3)

        # Label contour value along bottom-left of ellipse
        lx, ly = to_px(pk_x - r * 0.72, pk_y - r * b_ratio * 0.70)
        if x0 + 80 <= lx <= x1 - 80 and y0 + 110 <= ly <= y1 - 70:
            d.text((lx, ly), levels_lbl[idx], font=get_font(16, "demibold"), fill=(*VIOLET[:3], a_i), anchor="mm")

    # Peak marker
    pk_px = to_px(pk_x, pk_y)
    d.ellipse([pk_px[0] - 6, pk_px[1] - 6, pk_px[0] + 6, pk_px[1] + 6], fill=(*GOLD[:3], a_i))
    d.text((pk_px[0] + 14, pk_px[1] - 14), "Peak", font=get_font(18, "bold"), fill=(*GOLD[:3], a_i), anchor="lm")

    # Chosen point P on contour r = 1.24 at angle -135 deg (bottom-left of peak)
    r_p = 1.24
    phi = math.radians(-135.0)
    px_m = pk_x + r_p * math.cos(phi)
    py_m = pk_y + r_p * b_ratio * math.sin(phi)
    p_px = to_px(px_m, py_m)

    # Tangent and inward normal (toward peak = gradient!)
    tx = -r_p * math.sin(phi)
    ty = r_p * b_ratio * math.cos(phi)
    t_len = math.hypot(tx, ty)
    tx_u, ty_u = tx / t_len, ty / t_len
    # Inward perpendicular normal in math coords: (-(-ty_u), -tx_u)? Let's compute exact gradient of ((x-pk_x)^2 + ((y-pk_y)/b)^2) negated!
    nx_raw = -(px_m - pk_x)
    ny_raw = -(py_m - pk_y) / (b_ratio ** 2)
    n_len = math.hypot(nx_raw, ny_raw)
    nx_u, ny_u = nx_raw / n_len, ny_raw / n_len

    # Tangent line through P
    t_seg_a = to_px(px_m - 0.78 * tx_u, py_m - 0.78 * ty_u)
    t_seg_b = to_px(px_m + 0.78 * tx_u, py_m + 0.78 * ty_u)
    d.line([t_seg_a, t_seg_b], fill=(*AMBER[:3], int(220 * alpha)), width=3)

    # 90-degree square marker between tangent and gradient
    a_perp = sub_progress(t, 127.0, 129.0) * alpha
    if a_perp > 0.05:
        s_sq = 0.16
        sq1 = to_px(px_m + s_sq * tx_u, py_m + s_sq * ty_u)
        sq2 = to_px(px_m + s_sq * tx_u + s_sq * nx_u, py_m + s_sq * ty_u + s_sq * ny_u)
        sq3 = to_px(px_m + s_sq * nx_u, py_m + s_sq * ny_u)
        d.line([sq1, sq2, sq3], fill=(*GOLD[:3], int(255 * a_perp)), width=3)

    # Primary gradient arrow at P
    g_end = to_px(px_m + 0.85 * nx_u, py_m + 0.85 * ny_u)
    draw_arrow_2d(d, p_px, g_end, (*CYAN[:3], a_i), width=6, head_len=20)

    # 132.6 - 140.5: Sliding bead moving along the contour curve (same height!)
    a_bead = sub_progress(t, 132.6, 133.6) * (1.0 - sub_progress(t, 140.5, 141.5)) * alpha
    if a_bead > 0.02:
        phi_slide = phi + 0.65 * math.sin((t - 133.2) * 1.45) - 0.35
        bx_m = pk_x + r_p * math.cos(phi_slide)
        by_m = pk_y + r_p * b_ratio * math.sin(phi_slide)
        b_px = to_px(bx_m, by_m)
        d.ellipse([b_px[0] - 11, b_px[1] - 11, b_px[0] + 11, b_px[1] + 11], fill=(*EMERALD[:3], int(255 * a_bead)), outline=(255, 255, 255, int(255 * a_bead)), width=2)
        d.text((b_px[0] - 16, b_px[1] - 14), "z = 1.70 (const)", font=get_font(18, "bold"), fill=(*EMERALD[:3], int(255 * a_bead)), anchor="rm")

    # 140.8+: Radial orthogonal gradient vector field crossing contours
    a_field = sub_progress(t, 140.8, 143.2) * alpha
    if a_field > 0.02:
        for r_f in [0.82, 1.68]:
            for ang_deg in range(0, 360, 45):
                a_f = math.radians(ang_deg)
                fx = pk_x + r_f * math.cos(a_f)
                fy = pk_y + r_f * b_ratio * math.sin(a_f)
                if abs(fx - px_m) < 0.25 and abs(fy - py_m) < 0.25:
                    continue
                gx_f = -(fx - pk_x)
                gy_f = -(fy - pk_y) / (b_ratio ** 2)
                gm = math.hypot(gx_f, gy_f) + 1e-6
                p0_f = to_px(fx, fy)
                p1_f = to_px(fx + 0.34 * a_field * gx_f / gm, fy + 0.34 * a_field * gy_f / gm)
                if x0 + 65 <= p0_f[0] <= x1 - 65 and y0 + 100 <= p0_f[1] <= y1 - 65:
                    draw_arrow_2d(d, p0_f, p1_f, (*CYAN[:3], int(185 * a_field)), width=3, head_len=11)

    # Point P marker
    d.ellipse([p_px[0] - 9, p_px[1] - 9, p_px[0] + 9, p_px[1] + 9], fill=(*GOLD[:3], a_i), outline=(255, 255, 255, a_i), width=2)
    d.text((p_px[0] - 24, p_px[1] + 22), "P", font=get_font(22, "bold"), fill=(*GOLD[:3], a_i), anchor="mm")

    out = Image.alpha_composite(canvas.convert("RGBA"), layer).convert("RGB")
    canvas.paste(out)
    paste_tex(canvas, r"$\nabla f$", int(g_end[0]) + 24, int(g_end[1]) - 16, fontsize=24, color=CYAN, alpha=alpha, anchor="center")
    draw_text_alpha(canvas, (cx, y0 + 52), "Top-Down Contour Map  (Level Curves f(x, y) = c)", get_font(25, "bold"), color=TEXT, alpha=alpha, anchor="mm")


def render_scene3(t: float) -> Image.Image:
    """Scene 3 (96.80 -> 146.50s): Surface directional rates -> Max rate along ∇f -> Contour 90° orthogonality."""
    canvas = new_canvas()
    fade_in = sub_progress(t, 96.80, 97.50)
    fade_out = 1.0 - sub_progress(t, 145.80, 146.50)
    master = fade_in * fade_out

    left_box = (64, 102, 1070, 1018)
    draw_rounded_card(canvas, left_box, radius=24, alpha=0.96 * master)

    # Transition from 3D surface (96.8 - 126.2) to top-down 2D contour plot (125.0 - 146.5)
    a_3d = (1.0 - sub_progress(t, 125.0, 126.8)) * master
    a_2d = sub_progress(t, 125.2, 127.0) * master

    x0, y0 = -0.35, -0.25
    z0 = float(hill_f(x0, y0))
    gx, gy = hill_grad(x0, y0)
    gmag = math.hypot(gx, gy)
    grad_ang = math.atan2(gy, gx)

    # Candidate angle theta relative to gradient:
    # 103.4 - 116.2: smoothly rotates through different angles; 116.2+: aligns smoothly to theta = 0 (steepest!)
    if t < 103.4:
        theta = 0.5
    elif t < 116.2:
        theta = (t - 103.4) * 1.15  # sweeps ~14.7 rad (~2.3 rotations)
    else:
        # Ease smoothly to nearest multiple of 2*pi (theta_rel = 0)
        cur_end = (116.2 - 103.4) * 1.15
        target = round(cur_end / (2.0 * math.pi)) * 2.0 * math.pi
        theta = lerp(cur_end, target, sub_progress(t, 116.2, 118.2, ease_in_out_cubic))

    u_ang = grad_ang + theta
    ux, uy = math.cos(u_ang), math.sin(u_ang)
    dir_deriv = gx * ux + gy * uy  # exact directional derivative

    if a_3d > 0.02:
        # Tilt camera upward smoothly as we approach contour view at 124.2..126.8
        elev = 28.0 + 52.0 * sub_progress(t, 124.2, 126.8, ease_in_out_cubic)
        az = -50.0 + 8.0 * math.sin((t - 96.8) * 0.15)
        cam = Camera3D(azim_deg=az, elev_deg=elev, scale=142.0, center=(555, 670), z_scale=1.10)
        d = ImageDraw.Draw(canvas)
        draw_3d_base_grid(d, cam, xlim=(-2.0, 2.0), ylim=(-2.0, 2.0), z0=0.0, step=1.0)
        draw_3d_axes(d, cam, xlim=(-2.2, 2.4), ylim=(-2.2, 2.4), zlim=(0.0, 2.9), labels=("x", "y", "f(x, y)"))

        items = [
            {"type": "line", "pts": [(x0, y0, 0.0), (x0, y0, z0)], "color": GOLD, "width": 3, "dashed": True, "alpha": a_3d},
            {"type": "point", "p": (x0, y0, z0 + 0.05), "color": GOLD, "radius": 9, "label": "P", "label_offset": (-32, -16), "alpha": a_3d},
        ]

        # Base candidate ring around P on surface
        if t >= 103.0:
            a_ring = sub_progress(t, 103.0, 104.5) * a_3d
            ring_pts = []
            for k in range(49):
                ang = 2.0 * math.pi * k / 48.0
                rx, ry = x0 + 0.72 * math.cos(ang), y0 + 0.72 * math.sin(ang)
                ring_pts.append((rx, ry, float(hill_f(rx, ry)) + 0.04))
            items.append({"type": "line", "pts": ring_pts, "color": DIM, "width": 2, "alpha": 0.7 * a_ring})

            # Rotating tangent vector on surface
            u_color = CYAN if t >= 118.0 else (EMERALD if dir_deriv > 0.15 else (ROSE if dir_deriv < -0.15 else VIOLET))
            pts_u = [(x0 + ux * s, y0 + uy * s, float(hill_f(x0 + ux * s, y0 + uy * s)) + 0.05) for s in np.linspace(0, 0.88, 12)]
            items.append({"type": "line", "pts": pts_u, "color": u_color, "width": 6, "alpha": a_ring})
            items.append({"type": "arrow", "p0": pts_u[-2], "p1": pts_u[-1], "color": u_color, "width": 6, "label": r"$\nabla f$" if t >= 118.0 else r"$\hat{u}$", "is_tex": True, "label_offset": (40, -18), "alpha": a_ring})

        # Persistent cyan gradient shadow arrow on base plane once gradient is highlighted
        a_g3d = sub_progress(t, 116.2, 118.2) * a_3d
        if a_g3d > 0.02:
            gx_u, gy_u = math.cos(grad_ang), math.sin(grad_ang)
            items.append({"type": "arrow", "p0": (x0, y0, 0.03), "p1": (x0 + 1.0 * gx_u, y0 + 1.0 * gy_u, 0.03), "color": CYAN, "width": 6, "label": r"$\nabla f$", "is_tex": True, "label_offset": (38, 16), "alpha": a_g3d})

        render_surface_and_items(canvas, cam, _HX, _HY, _HZ, z_range=(0.1, 2.8), contour_levels=[0.6, 1.0, 1.4, 1.8, 2.2, 2.6], alpha=0.92 * a_3d, extra_items=items)
        draw_text_alpha(canvas, (567, 146), "Rates of Change Across Directions on f(x, y)", get_font(26, "bold"), color=TEXT, alpha=a_3d, anchor="mm")

    if a_2d > 0.02:
        _draw_2d_contour_panel(canvas, left_box, t, a_2d)

    # Right Explanatory Panel (1100..1856)
    draw_rounded_card(canvas, (1100, 102, 1856, 1018), radius=24, alpha=0.96 * master)

    if t < 125.5:
        a_p1 = (1.0 - sub_progress(t, 124.2, 125.5)) * master
        draw_text_alpha(canvas, (1145, 148), "WHY IS THE GRADIENT STEEPEST?", get_font(20, "bold"), color=CYAN, alpha=a_p1)
        draw_text_alpha(canvas, (1145, 190), "Directional Derivative", get_font(36, "bold"), color=TEXT, alpha=a_p1)

        # Polar dial showing angle theta between ∇f and u_hat
        draw_rounded_card(canvas, (1145, 255, 1810, 560), radius=18, fill=(19, 28, 52), outline=BORDER, alpha=a_p1)
        dial_cx, dial_cy = 1272, 405
        d = ImageDraw.Draw(canvas)
        d.ellipse([dial_cx - 95, dial_cy - 95, dial_cx + 95, dial_cy + 95], outline=BORDER, width=2)
        # Fixed ∇f pointing at -35° (screen coords up-right)
        g_scr_ang = -math.radians(35.0)
        g_tip = (dial_cx + 96 * math.cos(g_scr_ang), dial_cy + 96 * math.sin(g_scr_ang))
        draw_arrow_2d(d, (dial_cx, dial_cy), g_tip, CYAN, width=6, head_len=16)
        paste_tex(canvas, r"$\nabla f$", int(g_tip[0]) + 18, int(g_tip[1]) - 10, fontsize=18, color=CYAN, alpha=a_p1, anchor="center")

        # Rotating u_hat
        u_scr_ang = g_scr_ang - theta
        u_tip = (dial_cx + 76 * math.cos(u_scr_ang), dial_cy + 76 * math.sin(u_scr_ang))
        u_col = CYAN if t >= 118.0 else GOLD
        draw_arrow_2d(d, (dial_cx, dial_cy), u_tip, u_col, width=5, head_len=14)
        paste_tex(canvas, r"$\hat{u}$", int(u_tip[0]) + 18 * math.cos(u_scr_ang), int(u_tip[1]) + 18 * math.sin(u_scr_ang), fontsize=17, color=u_col, alpha=a_p1, anchor="center")

        # Formula + Live cosine gauge on right side of dial
        paste_tex(canvas, r"$D_{\hat{u}}f = \nabla f \cdot \hat{u} = \|\nabla f\|\cos\theta$", 1592, 326, fontsize=16, color=TEXT, alpha=a_p1, anchor="center")
        cos_val = math.cos(theta)
        deg_rel = (math.degrees(theta) % 360.0)
        if deg_rel > 180.0:
            deg_rel -= 360.0
        draw_text_alpha(canvas, (1592, 398), f"θ = {abs(deg_rel):5.1f}°", get_font(26, "bold", mono=True), color=GOLD, alpha=a_p1, anchor="mm")
        draw_text_alpha(canvas, (1592, 444), f"cos θ = {cos_val:+.2f}", get_font(26, "bold", mono=True), color=CYAN if cos_val > 0.96 else (EMERALD if cos_val > 0 else ROSE), alpha=a_p1, anchor="mm")
        status = "MAXIMUM ASCENT!" if cos_val > 0.97 and t >= 116.5 else ("Uphill (0 < cos θ < 1)" if cos_val > 0.08 else ("Downhill (cos θ < 0)" if cos_val < -0.08 else "Level (cos θ ≈ 0)"))
        draw_text_alpha(canvas, (1592, 498), status, get_font(21, "bold"), color=CYAN if cos_val > 0.97 and t >= 116.5 else TEXT, alpha=a_p1, anchor="mm")

        # Cosine wave card (bottom-right)
        draw_rounded_card(canvas, (1145, 585, 1810, 975), radius=18, fill=(19, 28, 52), outline=CYAN if t >= 116.5 else BORDER, outline_width=2 if t >= 116.5 else 1, alpha=a_p1)
        draw_text_alpha(canvas, (1180, 614), "Rate of Change vs. Angle θ", get_font(22, "bold"), color=TEXT, alpha=a_p1)
        gx0, gx1, gy_mid = 1210, 1755, 765
        d.line([(gx0, gy_mid), (gx1, gy_mid)], fill=DIM, width=2)
        d.line([(gx0, 665), (gx0, 865)], fill=DIM, width=2)
        wave_pts = []
        for k in range(91):
            ang = -math.pi + (2.0 * math.pi * k / 90.0)
            wx = gx0 + (k / 90.0) * (gx1 - gx0)
            wy = gy_mid - 78 * math.cos(ang)
            wave_pts.append((wx, wy))
        d.line(wave_pts, fill=VIOLET, width=4)
        # Moving dot at current angle wrapped to [-pi, pi]
        ang_wrap = math.radians(deg_rel)
        dot_x = gx0 + ((ang_wrap + math.pi) / (2.0 * math.pi)) * (gx1 - gx0)
        dot_y = gy_mid - 78 * math.cos(ang_wrap)
        d.ellipse([dot_x - 9, dot_y - 9, dot_x + 9, dot_y + 9], fill=CYAN, outline=TEXT, width=2)
        draw_text_alpha(canvas, ((gx0 + gx1) // 2, 668), "Peak at θ = 0° (along gradient)", get_font(18, "bold"), color=CYAN, alpha=a_p1, anchor="mm")
        draw_text_alpha(canvas, (1478, 922), "Maximum increase occurs when û aligns with gradient (cos 0° = 1)", get_font(19, "demibold"), color=GOLD if t >= 116.5 else MUTED, alpha=a_p1, anchor="mm")
    else:
        # Phase B right panel: Contour Perpendicularity Explanation (125.2 - 146.5)
        a_p2 = sub_progress(t, 125.2, 126.8) * master
        draw_text_alpha(canvas, (1145, 148), "CONTOUR GEOMETRY", get_font(20, "bold"), color=VIOLET, alpha=a_p2)
        draw_text_alpha(canvas, (1145, 190), "Why Gradient Is 90° to Contours", get_font(34, "bold"), color=TEXT, alpha=a_p2)

        cards = [
            (126.0, "1. Contour = Constant Height", r"$f(x, y) = c \;\;\mathrm{(constant\ on\ curve)}$", VIOLET,
             "Every point on a contour line sits at the exact same elevation."),
            (132.5, "2. Zero Change Along Contour", r"$\Delta f = 0 \;\Rightarrow\; \nabla f \cdot \hat{t} = 0$", EMERALD,
             "Moving along the tangent t̂ produces zero height change."),
            (139.0, "3. Dot Product Zero ⇒ 90° Angle", r"$\nabla f \cdot \hat{t} = \|\nabla f\|\cos 90^\circ = 0 \;\Rightarrow\; \nabla f \perp \hat{t}$", CYAN,
             "The gradient crosses contours at right angles toward steeper ascent."),
        ]
        for idx, (ts, title, tex, col, desc) in enumerate(cards):
            ac = sub_progress(t, ts, ts + 1.2) * a_p2
            if ac > 0.02:
                y_top = 265 + idx * 235
                draw_rounded_card(canvas, (1145, y_top, 1810, y_top + 205), radius=18, fill=(19, 28, 52), outline=col, outline_width=2, alpha=ac)
                draw_text_alpha(canvas, (1178, y_top + 26), title, get_font(24, "bold"), color=col, alpha=ac)
                paste_tex(canvas, tex, 1478, y_top + 106, fontsize=17, color=TEXT, alpha=ac, anchor="center")
                draw_text_alpha(canvas, (1478, y_top + 168), desc, get_font(19, "regular"), color=MUTED, alpha=ac, anchor="mm")

    draw_scene_chrome(canvas, t, TOTAL_DURATION, "SCENE 03 / 06", "Visual Intuition & Contours")
    return canvas
