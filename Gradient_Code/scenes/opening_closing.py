"""Opening title frame (0:00-0:08.8) and Closing summary & credits frame (4:41.2-4:52.0)."""

import math
import numpy as np
from PIL import Image, ImageDraw
from styles import (
    WIDTH, HEIGHT, CYAN, AMBER, EMERALD, VIOLET, ROSE, GOLD, TEXT, MUTED, DIM,
    BG_CARD, BORDER,
    get_font, new_canvas, draw_rounded_card, draw_text_alpha, paste_tex,
    draw_arrow_2d, draw_scene_chrome, sub_progress, ease_out_cubic, ease_in_out_cubic, clamp,
)
from timeline import COURSE_TITLE, STUDENT_NAME, STUDENT_ID, TOTAL_DURATION


def _draw_bg_contours(canvas: Image.Image, phase: float, alpha: float = 0.25):
    """Draw subtle animated concentric contour ellipses in the background."""
    if alpha <= 0.01:
        return
    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    cx, cy = int(WIDTH * 0.78), int(HEIGHT * 0.48)
    for k in range(1, 9):
        rx = 95 * k + 12 * math.sin(phase + k * 0.5)
        ry = 62 * k + 8 * math.cos(phase + k * 0.5)
        col = (*VIOLET[:3], int(round(255 * alpha * max(0.2, 1.0 - k * 0.09))))
        d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], outline=col, width=2)
    # Draw glowing gradient vectors perpendicular to contour ellipses
    for ang_deg in [25, 135, 225, 325]:
        ang = math.radians(ang_deg)
        r0, r1 = 190, 295
        x0, y0 = cx + r0 * math.cos(ang) * 1.45, cy + r0 * math.sin(ang) * 0.95
        x1, y1 = cx + r1 * math.cos(ang) * 1.45, cy + r1 * math.sin(ang) * 0.95
        draw_arrow_2d(d, (x0, y0), (x1, y1), (*CYAN[:3], int(round(255 * alpha * 1.3))), width=4, head_len=14)
    out = Image.alpha_composite(canvas.convert("RGBA"), layer).convert("RGB")
    canvas.paste(out)


def render_opening(t: float) -> Image.Image:
    """Opening Frame (0.00 -> 8.80s): Title, Concept, Student Name & ID."""
    canvas = new_canvas()
    # Smooth fade-in from 0.2s and gentle fade-out into Scene 1 at 8.2..8.8s
    master = sub_progress(t, 0.15, 1.2) * (1.0 - sub_progress(t, 8.20, 8.80))
    _draw_bg_contours(canvas, phase=t * 0.6, alpha=0.22 * master)

    # Left hero card
    draw_rounded_card(canvas, (110, 175, 1270, 895), radius=28, fill=(15, 22, 42), outline=BORDER, alpha=0.92 * master, accent_bar=CYAN)

    a_course = sub_progress(t, 0.4, 1.3) * master
    a_title = sub_progress(t, 0.8, 1.8) * master
    a_sub = sub_progress(t, 1.4, 2.5) * master
    a_meta = sub_progress(t, 2.0, 3.2) * master

    f_course = get_font(22, "demibold")
    f_title = get_font(74, "bold")
    f_sub = get_font(34, "medium")
    f_meta_lbl = get_font(20, "medium")
    f_meta_val = get_font(30, "demibold")

    draw_text_alpha(canvas, (170, 240), COURSE_TITLE.upper(), f_course, color=CYAN, alpha=a_course)
    draw_text_alpha(canvas, (170, 320), "THE GRADIENT", f_title, color=TEXT, alpha=a_title)
    draw_text_alpha(canvas, (170, 418), "Understanding the Direction of Steepest Ascent", f_sub, color=MUTED, alpha=a_sub)

    # Concept formula badge
    if a_sub > 0.05:
        draw_rounded_card(canvas, (170, 505, 760, 625), radius=18, fill=(20, 31, 58), outline=(56, 189, 248), outline_width=2, alpha=0.95 * a_sub)
        paste_tex(canvas, r"$\nabla f(x, y) = \left( \frac{\partial f}{\partial x}, \; \frac{\partial f}{\partial y} \right)$", 465, 565, fontsize=27, color=CYAN, alpha=a_sub, anchor="center")

    # Student metadata footer inside card
    if a_meta > 0.05:
        draw_rounded_card(canvas, (170, 695, 1215, 825), radius=16, fill=(19, 28, 52), outline=BORDER, outline_width=1, alpha=0.95 * a_meta)
        draw_text_alpha(canvas, (205, 725), "PRESENTED BY", f_meta_lbl, color=MUTED, alpha=a_meta)
        draw_text_alpha(canvas, (205, 760), STUDENT_NAME, f_meta_val, color=TEXT, alpha=a_meta)

        draw_text_alpha(canvas, (560, 725), "STUDENT ID", f_meta_lbl, color=MUTED, alpha=a_meta)
        draw_text_alpha(canvas, (560, 760), STUDENT_ID, get_font(30, "bold", mono=True), color=GOLD, alpha=a_meta)

        draw_text_alpha(canvas, (860, 725), "TOPIC AREA", f_meta_lbl, color=MUTED, alpha=a_meta)
        draw_text_alpha(canvas, (860, 760), "Multivariable Calculus", f_meta_val, color=EMERALD, alpha=a_meta)

    # Right emblem card: animated nabla vector compass
    a_emblem = sub_progress(t, 1.1, 2.4) * master
    if a_emblem > 0.05:
        draw_rounded_card(canvas, (1320, 260, 1810, 810), radius=28, fill=(15, 22, 42), outline=BORDER, alpha=0.9 * a_emblem)
        layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        cx, cy = 1565, 515
        for k, rad in enumerate([75, 130, 185]):
            d.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], outline=(*VIOLET[:3], int(190 * a_emblem)), width=2)
        ang = -math.pi / 4 + 0.15 * math.sin(t * 1.1)
        tip = (cx + 155 * math.cos(ang), cy + 155 * math.sin(ang))
        draw_arrow_2d(d, (cx, cy), tip, (*CYAN[:3], int(255 * a_emblem)), width=7, head_len=22)
        d.ellipse([cx - 9, cy - 9, cx + 9, cy + 9], fill=(*GOLD[:3], int(255 * a_emblem)), outline=(255, 255, 255, int(255 * a_emblem)), width=2)
        canvas = Image.alpha_composite(canvas.convert("RGBA"), layer).convert("RGB")
        paste_tex(canvas, r"$\nabla f$", int(tip[0]) + 28, int(tip[1]) - 18, fontsize=28, color=CYAN, alpha=a_emblem, anchor="center")
        draw_text_alpha(canvas, (1565, 755), "Steepest Ascent Direction", get_font(22, "demibold"), color=MUTED, alpha=a_emblem, anchor="mm")

    draw_scene_chrome(canvas, t, TOTAL_DURATION, "OPENING", "Title & Introduction")
    return canvas


def render_closing(t: float) -> Image.Image:
    """Closing Frame (281.20 -> 292.00s): 3 Core Takeaways, References, Music Credit & AI Disclosure."""
    canvas = new_canvas()
    a_in = sub_progress(t, 281.20, 282.30)
    _draw_bg_contours(canvas, phase=t * 0.4, alpha=0.14 * a_in)

    f_hdr = get_font(44, "bold")
    f_sub = get_font(22, "medium")
    f_card_t = get_font(24, "bold")
    f_card_b = get_font(20, "regular")
    f_small_b = get_font(19, "regular")

    draw_text_alpha(canvas, (960, 128), "THANK YOU FOR WATCHING", f_hdr, color=TEXT, alpha=a_in, anchor="mm")
    draw_text_alpha(canvas, (960, 172), f"{STUDENT_NAME}  •  ID: {STUDENT_ID}  •  {COURSE_TITLE}", f_sub, color=CYAN, alpha=a_in, anchor="mm")

    # Top row: 3 Core Takeaways recap
    takeaways = [
        ("1. Vector of Partial Derivatives", r"$\nabla f = \left(\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}\right)$", "Encodes x and y rates of change", CYAN),
        ("2. Steepest Ascent Direction", r"$D_{\hat{u}}f = \nabla f \cdot \hat{u} = \|\nabla f\|\cos\theta$", "Peaks at θ = 0° (along gradient)", GOLD),
        ("3. Perpendicular to Contours", r"$\nabla f \perp \mathrm{level\ curve}\ f(x,y)=c$", "Crosses contours toward higher z", VIOLET),
    ]
    for idx, (title, tex, desc, col) in enumerate(takeaways):
        x0 = 80 + idx * 600
        x1 = x0 + 560
        a_c = sub_progress(t, 281.4 + idx * 0.25, 282.4 + idx * 0.25)
        draw_rounded_card(canvas, (x0, 225, x1, 455), radius=20, fill=BG_CARD, outline=col, outline_width=2, alpha=0.95 * a_c)
        draw_text_alpha(canvas, (x0 + 28, 256), title, f_card_t, color=col, alpha=a_c)
        paste_tex(canvas, tex, (x0 + x1) // 2, 342, fontsize=21, color=TEXT, alpha=a_c, anchor="center")
        draw_text_alpha(canvas, ((x0 + x1) // 2, 416), desc, f_card_b, color=MUTED, alpha=a_c, anchor="mm")

    # Bottom row: References & Credits + AI Disclosure (Required by Rubric)
    a_bot = sub_progress(t, 282.3, 283.6)
    draw_rounded_card(canvas, (80, 495, 940, 995), radius=22, fill=(15, 22, 42), outline=BORDER, alpha=0.96 * a_bot, accent_bar=EMERALD)
    draw_text_alpha(canvas, (122, 526), "References & Audio Credits", get_font(26, "bold"), color=EMERALD, alpha=a_bot)
    refs = [
        "• James Stewart, Calculus: Early Transcendentals (Partial Derivatives & Gradients)",
        "• 3Blue1Brown & Khan Academy: Geometric Intuition of Directional Derivatives",
        "• Concept & Voice-over Narration: Krisha Jogiya (E25B070935)",
        "• Background Music: Original subtle ambient harmonic pad synthesized",
        "  procedurally in Python (NumPy/SciPy) and ducked beneath voice-over.",
        "• Animation Source Code: Custom Python scripts (NumPy, Matplotlib, Pillow, FFmpeg).",
    ]
    for k, line in enumerate(refs):
        draw_text_alpha(canvas, (122, 588 + k * 56), line, f_small_b, color=TEXT if k in (2, 3) else MUTED, alpha=a_bot)

    draw_rounded_card(canvas, (980, 495, 1840, 995), radius=22, fill=(15, 22, 42), outline=BORDER, alpha=0.96 * a_bot, accent_bar=AMBER)
    draw_text_alpha(canvas, (1022, 526), "AI & Tooling Disclosure", get_font(26, "bold"), color=AMBER, alpha=a_bot)
    disc = [
        "In accordance with course originality guidelines:",
        "• Voice-over narration was recorded directly by Krisha Jogiya.",
        "• Mathematical derivations, worked example f(x, y) = x² + y² at (1, 1),",
        "  and unit gradient calculations were verified independently.",
        "• AI assistance (Antigravity / Gemini) was used to help refine the",
        "  storyboard structure, sync animation timestamps, and generate the",
        "  Python visualization code (NumPy / Matplotlib / Pillow / FFmpeg).",
    ]
    for k, line in enumerate(disc):
        draw_text_alpha(canvas, (1022, 588 + k * 52), line, f_small_b, color=TEXT if k == 1 else MUTED, alpha=a_bot)

    draw_scene_chrome(canvas, t, TOTAL_DURATION, "CLOSING", "Summary, References & Credits")
    return canvas
