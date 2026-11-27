"""Shared visual design tokens, typography, STIX math renderer, and drawing helpers."""

from functools import lru_cache
import io
import math
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WIDTH = 1920
HEIGHT = 1080

# Consistent color coding across all scenes (Dark Slate Math Studio)
BG = (11, 16, 32)            # #0B1020 Midnight slate canvas
BG_DEEP = (7, 11, 22)        # #070B16 Deep gradient edge
BG_CARD = (18, 26, 48)       # #121A30 Elevated card fill
BG_CARD_ALT = (24, 35, 64)   # #182340 Highlighted panel
BORDER = (44, 62, 102)       # #2C3E66 Subtle card stroke
BORDER_SOFT = (28, 40, 68)   # #1C2844 Grid / rule lines

TEXT = (248, 250, 252)       # #F8FAFC Primary crisp ink
MUTED = (148, 163, 184)      # #94A3B8 Secondary slate ink
DIM = (90, 106, 134)         # #5A6A86 Subtle annotations

CYAN = (56, 189, 248)        # #38BDF8 Gradient vector ∇f / Steepest ascent
AMBER = (245, 158, 11)       # #F59E0B x-component ∂f/∂x
EMERALD = (52, 211, 153)     # #34D399 y-component ∂f/∂y
VIOLET = (167, 139, 250)     # #A78BFA Contour lines / Level curves
ROSE = (251, 113, 133)       # #FB7185 Unit vector û / Gradient descent -∇f
GOLD = (250, 204, 21)        # #FACC15 Key takeaway / peak highlight

plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "STIXGeneral"

AVENIR_PATH = "/System/Library/Fonts/Avenir Next.ttc"
MENLO_PATH = "/System/Library/Fonts/Menlo.ttc"
HELVETICA_PATH = "/System/Library/Fonts/HelveticaNeue.ttc"

# Avenir Next TTC face indices verified on macOS:
# 0: Bold, 2: Demi Bold, 4: Italic, 5: Medium, 7: Regular
_FONT_CACHE = {}


def get_font(size: int, weight: str = "regular", mono: bool = False):
    """Load cached Pillow font with crisp weight selection."""
    key = (size, weight, mono)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    if mono and os.path.exists(MENLO_PATH):
        idx = 1 if weight == "bold" else 0
        f = ImageFont.truetype(MENLO_PATH, size, index=idx)
    elif os.path.exists(AVENIR_PATH):
        idx_map = {
            "bold": 0,
            "demibold": 2,
            "semibold": 2,
            "italic": 4,
            "medium": 5,
            "regular": 7,
        }
        f = ImageFont.truetype(AVENIR_PATH, size, index=idx_map.get(weight, 7))
    elif os.path.exists(HELVETICA_PATH):
        f = ImageFont.truetype(HELVETICA_PATH, size)
    else:
        f = ImageFont.load_default()
    _FONT_CACHE[key] = f
    return f


def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*rgb[:3])


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def lerp_color(c1, c2, t: float):
    t = clamp(t)
    return tuple(int(round(c1[i] + (c2[i] - c1[i]) * t)) for i in range(len(c1)))


def ease_out_cubic(t: float) -> float:
    t = clamp(t)
    return 1.0 - (1.0 - t) ** 3


def ease_in_out_cubic(t: float) -> float:
    t = clamp(t)
    if t < 0.5:
        return 4.0 * t * t * t
    return 1.0 - ((-2.0 * t + 2.0) ** 3) / 2.0


def ease_out_back(t: float) -> float:
    t = clamp(t)
    c1 = 1.70158
    c3 = c1 + 1.0
    return 1.0 + c3 * ((t - 1.0) ** 3) + c1 * ((t - 1.0) ** 2)


def sub_progress(t: float, t0: float, t1: float, ease=ease_out_cubic) -> float:
    """Normalized eased progress in [0, 1] over [t0, t1]."""
    if t1 <= t0:
        return 1.0 if t >= t0 else 0.0
    return ease(clamp((t - t0) / (t1 - t0)))


def fade_alpha(t: float, t_in0: float, t_in1: float, t_out0: float = 1e9, t_out1: float = 1e9) -> float:
    """Smooth fade-in and optional fade-out alpha multiplier."""
    a_in = sub_progress(t, t_in0, t_in1, ease_out_cubic)
    if t >= t_out0:
        a_out = 1.0 - sub_progress(t, t_out0, t_out1, ease_out_cubic)
        return a_in * a_out
    return a_in


@lru_cache(maxsize=512)
def render_tex_rgba(tex: str, fontsize: int = 24, color_rgb=(248, 250, 252), dpi: int = 160) -> Image.Image:
    """Render a LaTeX/mathtext expression using Matplotlib STIX onto a tightly cropped RGBA image."""
    # Measure text extent cleanly inside a roomy scratch figure so glyphs never clip
    fig_m = plt.figure(figsize=(14, 4), dpi=dpi)
    fig_m.patch.set_alpha(0.0)
    txt = fig_m.text(0.5, 0.5, tex, fontsize=fontsize, color=rgb_to_hex(color_rgb), va="center", ha="center")
    fig_m.canvas.draw()
    buf = io.BytesIO()
    fig_m.savefig(buf, format="png", dpi=dpi, transparent=True)
    plt.close(fig_m)
    buf.seek(0)
    im = Image.open(buf).convert("RGBA")
    alpha_box = im.getbbox()
    if alpha_box:
        pad = 6
        x0 = max(0, alpha_box[0] - pad)
        y0 = max(0, alpha_box[1] - pad)
        x1 = min(im.width, alpha_box[2] + pad)
        y1 = min(im.height, alpha_box[3] + pad)
        im = im.crop((x0, y0, x1, y1))
    else:
        im = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
    return im


def paste_rgba(canvas: Image.Image, overlay: Image.Image, x: int, y: int, alpha: float = 1.0, anchor: str = "tl"):
    """Composite an RGBA image onto an RGB/RGBA canvas with optional opacity and anchor."""
    if alpha <= 0.01:
        return
    if alpha < 0.999:
        r, g, b, a = overlay.split()
        a = a.point(lambda p: int(p * alpha))
        overlay = Image.merge("RGBA", (r, g, b, a))
    w, h = overlay.size
    if anchor == "center":
        px, py = int(x - w / 2), int(y - h / 2)
    elif anchor == "cl":
        px, py = int(x), int(y - h / 2)
    elif anchor == "cr":
        px, py = int(x - w), int(y - h / 2)
    elif anchor == "tc":
        px, py = int(x - w / 2), int(y)
    else:
        px, py = int(x), int(y)
    canvas.paste(overlay, (px, py), overlay)


def paste_tex(canvas: Image.Image, tex: str, x: int, y: int, fontsize: int = 30, color=TEXT, alpha: float = 1.0, anchor: str = "tl"):
    """Render and composite cached math formula onto canvas."""
    if alpha <= 0.01:
        return
    im = render_tex_rgba(tex, fontsize=fontsize, color_rgb=tuple(color[:3]))
    paste_rgba(canvas, im, x, y, alpha=alpha, anchor=anchor)


@lru_cache(maxsize=4)
def get_base_backdrop() -> Image.Image:
    """Pre-render crisp dark slate background with subtle radial vignette and coordinate dot grid."""
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]
    cx, cy = WIDTH * 0.52, HEIGHT * 0.45
    dist = np.sqrt(((xx - cx) / (WIDTH * 0.65)) ** 2 + ((yy - cy) / (HEIGHT * 0.65)) ** 2)
    t = np.clip(dist, 0.0, 1.0)
    arr = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    for c in range(3):
        arr[:, :, c] = np.clip(BG[c] * (1.0 - 0.38 * t) + BG_DEEP[c] * (0.38 * t), 0, 255).astype(np.uint8)
    # Subtle faint dot grid every 48px
    for gx in range(48, WIDTH, 48):
        for gy in range(48, HEIGHT, 48):
            arr[gy:gy+2, gx:gx+2] = (23, 33, 58)
    return Image.fromarray(arr, mode="RGB")


def new_canvas() -> Image.Image:
    """Return a fresh copy of the 1920x1080 dark slate backdrop."""
    return get_base_backdrop().copy()


def draw_rounded_card(
    canvas: Image.Image,
    box,
    radius: int = 20,
    fill=BG_CARD,
    outline=BORDER,
    outline_width: int = 2,
    alpha: float = 1.0,
    accent_bar=None,
):
    """Draw an anti-aliased rounded card with optional left/top accent stripe."""
    if alpha <= 0.01:
        return
    x0, y0, x1, y1 = [int(v) for v in box]
    layer = Image.new("RGBA", (x1 - x0 + 4, y1 - y0 + 4), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    fill_rgba = (*fill[:3], int(round(238 * alpha)))
    out_rgba = (*outline[:3], int(round(255 * alpha)))
    d.rounded_rectangle([2, 2, x1 - x0 + 2, y1 - y0 + 2], radius=radius, fill=fill_rgba, outline=out_rgba, width=outline_width)
    if accent_bar is not None:
        bar_rgba = (*accent_bar[:3], int(round(255 * alpha)))
        d.rounded_rectangle([2, 2, 10, y1 - y0 + 2], radius=4, fill=bar_rgba)
    canvas.paste(layer, (x0 - 2, y0 - 2), layer)


def draw_text_alpha(
    canvas: Image.Image,
    pos,
    text: str,
    font,
    color=TEXT,
    alpha: float = 1.0,
    anchor: str = "la",
):
    """Draw anti-aliased text with alpha blending onto RGB canvas."""
    if alpha <= 0.01:
        return
    if alpha >= 0.995:
        d = ImageDraw.Draw(canvas)
        d.text(pos, text, font=font, fill=tuple(color[:3]), anchor=anchor)
        return
    # Blend via temporary RGBA crop around text bbox
    tmp_draw = ImageDraw.Draw(canvas)
    ix, iy = int(round(pos[0])), int(round(pos[1]))
    bbox = tmp_draw.textbbox((ix, iy), text, font=font, anchor=anchor)
    pad = 8
    x0 = int(max(0, math.floor(bbox[0]) - pad))
    y0 = int(max(0, math.floor(bbox[1]) - pad))
    x1 = int(min(WIDTH, math.ceil(bbox[2]) + pad))
    y1 = int(min(HEIGHT, math.ceil(bbox[3]) + pad))
    if x1 <= x0 or y1 <= y0:
        return
    crop = canvas.crop((x0, y0, x1, y1)).convert("RGBA")
    overlay = Image.new("RGBA", crop.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    od.text((ix - x0, iy - y0), text, font=font, fill=(*color[:3], int(round(255 * alpha))), anchor=anchor)
    composited = Image.alpha_composite(crop, overlay).convert("RGB")
    canvas.paste(composited, (x0, y0))


def draw_arrow_2d(draw: ImageDraw.ImageDraw, p0, p1, color, width: int = 5, head_len: float = 18.0, head_angle: float = 0.44):
    """Draw a clean geometric 2D vector arrow from p0 to p1."""
    x0, y0 = p0
    x1, y1 = p1
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    if length < 2.0:
        return
    ux, uy = dx / length, dy / length
    hl = min(head_len, length * 0.45)
    bx, by = x1 - ux * hl * 0.72, y1 - uy * hl * 0.72
    draw.line([(x0, y0), (bx, by)], fill=color, width=width)
    ang = math.atan2(dy, dx)
    p_left = (x1 - hl * math.cos(ang - head_angle), y1 - hl * math.sin(ang - head_angle))
    p_right = (x1 - hl * math.cos(ang + head_angle), y1 - hl * math.sin(ang + head_angle))
    draw.polygon([(x1, y1), p_left, p_right], fill=color)


def draw_scene_chrome(canvas: Image.Image, t: float, total_duration: float, badge: str, subtitle: str):
    """Render minimal top breadcrumb pill and thin bottom timeline bar."""
    f_badge = get_font(18, "bold")
    f_sub = get_font(20, "medium")
    # Top-left pill
    draw_rounded_card(canvas, (64, 30, 590, 74), radius=12, fill=(16, 23, 44), outline=BORDER_SOFT, outline_width=1, alpha=0.92)
    d = ImageDraw.Draw(canvas)
    d.ellipse([80, 46, 92, 58], fill=CYAN)
    d.text((104, 52), badge, font=f_badge, fill=CYAN, anchor="lm")
    d.text((255, 52), "•   " + subtitle, font=f_sub, fill=MUTED, anchor="lm")

    # Top-right subtle title mark
    f_right = get_font(18, "medium")
    d.text((WIDTH - 64, 52), "Gradient  |  M3 Internal Assessment", font=f_right, fill=DIM, anchor="rm")

    # Bottom progress bar
    bar_y = HEIGHT - 10
    d.rectangle([0, bar_y, WIDTH, HEIGHT], fill=(16, 24, 42))
    prog = clamp(t / total_duration)
    d.rectangle([0, bar_y, int(WIDTH * prog), HEIGHT], fill=CYAN)
