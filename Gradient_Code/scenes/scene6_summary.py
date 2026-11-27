"""Scene 6 (256.30s - 281.20s): Summary & Mountain Takeaway."""

import math
import numpy as np
from PIL import Image, ImageDraw
from styles import (
    WIDTH, HEIGHT, CYAN, AMBER, EMERALD, VIOLET, ROSE, GOLD, TEXT, MUTED, DIM,
    BG_CARD, BORDER,
    get_font, new_canvas, draw_rounded_card, draw_text_alpha, paste_tex,
    draw_arrow_2d, draw_scene_chrome, sub_progress, ease_out_cubic, ease_in_out_cubic, clamp, lerp,
)
from timeline import TOTAL_DURATION


def render_scene6(t: float) -> Image.Image:
    """Scene 6 (256.30 -> 281.20s): 3 Core Takeaways + Animated Mountain Climb Recall."""
    canvas = new_canvas()
    fade_in = sub_progress(t, 256.30, 257.00)
    fade_out = 1.0 - sub_progress(t, 280.60, 281.20)
    master = fade_in * fade_out

    draw_text_alpha(canvas, (960, 125), "CONCLUSION & RECAP", get_font(20, "bold"), color=CYAN, alpha=master, anchor="mm")
    draw_text_alpha(canvas, (960, 168), "Three Key Takeaways About the Gradient", get_font(42, "bold"), color=TEXT, alpha=master, anchor="mm")

    cards = [
        (258.5, "1. Vector of Partial Derivatives", r"$\nabla f = \left( \frac{\partial f}{\partial x}, \; \frac{\partial f}{\partial y} \right)$",
         "Combines rates of change in x and y into one vector.", CYAN),
        (262.4, "2. Steepest Increase Direction", r"$D_{\hat{u}}f \leq \|\nabla f\| \;\; (\mathrm{max\ at\ } \hat{u} \parallel \nabla f)$",
         "Points directly toward the fastest possible ascent.", GOLD),
        (266.8, "3. Orthogonal to Contour Lines", r"$\nabla f \perp \mathrm{contour}\;\; f(x, y) = c$",
         "Crosses level curves at 90° toward higher elevation.", VIOLET),
    ]

    for idx, (ts, title, tex, desc, col) in enumerate(cards):
        ac = sub_progress(t, ts, ts + 1.1) * master
        if ac > 0.02:
            x0 = 80 + idx * 600
            x1 = x0 + 560
            draw_rounded_card(canvas, (x0, 225, x1, 555), radius=22, fill=BG_CARD, outline=col, outline_width=2, alpha=0.96 * ac, accent_bar=col)
            draw_text_alpha(canvas, (x0 + 36, 260), title, get_font(25, "bold"), color=col, alpha=ac)
            paste_tex(canvas, tex, (x0 + x1) // 2, 378, fontsize=22, color=TEXT, alpha=ac, anchor="center")
            draw_text_alpha(canvas, ((x0 + x1) // 2, 488), desc, get_font(19, "regular"), color=MUTED, alpha=ac, anchor="mm")

    # Bottom Recall Panel: Stylized mountain ridge + glowing steepest-ascent climber (271.5+)
    a_mtn = sub_progress(t, 271.5, 273.0) * master
    if a_mtn > 0.02:
        draw_rounded_card(canvas, (80, 595, 1840, 1002), radius=24, fill=(14, 21, 40), outline=BORDER, alpha=0.96 * a_mtn)

        layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        a_i = int(round(255 * a_mtn))

        # Stylized layered mountain profile on left-center
        xs = np.linspace(140, 1140, 160)
        ys = 955 - 275 * np.exp(-((xs - 680) / 215.0) ** 2) - 95 * np.exp(-((xs - 410) / 145.0) ** 2)
        poly_pts = [(140, 965)] + list(zip(xs.tolist(), ys.tolist())) + [(1140, 965)]
        d.polygon(poly_pts, fill=(24, 42, 82, int(220 * a_mtn)))
        d.line(list(zip(xs.tolist(), ys.tolist())), fill=(*VIOLET[:3], a_i), width=4)

        # Elevation guide lines
        for y_lvl in [885, 815, 745]:
            d.line([(160, y_lvl), (1120, y_lvl)], fill=(45, 65, 110, int(110 * a_mtn)), width=1)

        # Animated hiker climbing straight up the steep east/west ridge along ∇f
        climb_p = sub_progress(t, 272.8, 279.5, ease_in_out_cubic)
        hx = lerp(495.0, 680.0, climb_p)
        hy = 955 - 275 * math.exp(-((hx - 680) / 215.0) ** 2) - 95 * math.exp(-((hx - 410) / 145.0) ** 2)

        # Trail behind hiker
        trail_xs = np.linspace(495.0, hx, 30)
        trail_ys = 955 - 275 * np.exp(-((trail_xs - 680) / 215.0) ** 2) - 95 * np.exp(-((trail_xs - 410) / 145.0) ** 2)
        if len(trail_xs) >= 2:
            d.line(list(zip(trail_xs.tolist(), trail_ys.tolist())), fill=(*CYAN[:3], a_i), width=6)

        # Arrow pointing ahead along steepest slope
        nx = min(680.0, hx + 55.0)
        ny = 955 - 275 * math.exp(-((nx - 680) / 215.0) ** 2) - 95 * np.exp(-((nx - 410) / 145.0) ** 2)
        if math.hypot(nx - hx, ny - hy) > 10:
            draw_arrow_2d(d, (hx, hy), (nx, ny - 4), (*CYAN[:3], a_i), width=6, head_len=16)

        d.ellipse([hx - 10, hy - 10, hx + 10, hy + 10], fill=(*GOLD[:3], a_i), outline=(255, 255, 255, a_i), width=2)
        # Summit flag
        d.line([(680, 680), (680, 636)], fill=(*GOLD[:3], a_i), width=3)
        d.polygon([(680, 636), (714, 648), (680, 660)], fill=(*GOLD[:3], a_i))

        canvas = Image.alpha_composite(canvas.convert("RGBA"), layer).convert("RGB")
        draw_text_alpha(canvas, (1470, 725), "Fastest Uphill Path", get_font(34, "bold"), color=TEXT, alpha=a_mtn, anchor="mm")
        paste_tex(canvas, r"$\mathrm{Direction} = \frac{\nabla f}{\|\nabla f\|}$", 1470, 815, fontsize=28, color=CYAN, alpha=a_mtn, anchor="center")
        draw_text_alpha(canvas, (1470, 905), "Always given by the Gradient vector", get_font(22, "medium"), color=MUTED, alpha=a_mtn, anchor="mm")

    draw_scene_chrome(canvas, t, TOTAL_DURATION, "SCENE 06 / 06", "Summary & Key Takeaways")
    return canvas
