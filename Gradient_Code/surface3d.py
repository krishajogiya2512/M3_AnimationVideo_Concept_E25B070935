"""Lightweight, deterministic 3D surface and vector renderer using NumPy + Pillow.

Uses a right-handed 3D camera rotation and unified back-to-front painter's sorting
across surface quads, level contours, and 3D vector primitives so there are
no Matplotlib mplot3d z-order clipping artifacts.
"""

import math
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from styles import CYAN, TEXT, MUTED, BORDER, get_font, clamp, render_tex_rgba, paste_rgba


class Camera3D:
    """Orthographic 3D-to-2D camera with azimuth, elevation, zoom, and viewport center."""

    def __init__(self, azim_deg: float = -48.0, elev_deg: float = 28.0, scale: float = 115.0, center=(540, 560), z_scale: float = 1.0):
        self.az = math.radians(azim_deg)
        self.el = math.radians(elev_deg)
        self.scale = scale
        self.cx, self.cy = center
        self.z_scale = z_scale
        self.cos_az, self.sin_az = math.cos(self.az), math.sin(self.az)
        self.cos_el, self.sin_el = math.cos(self.el), math.sin(self.el)

    def project(self, x, y, z):
        """Project world (x, y, z) arrays or scalars to (screen_x, screen_y, depth)."""
        z_s = np.asarray(z, dtype=np.float64) * self.z_scale
        x_a = np.asarray(x, dtype=np.float64)
        y_a = np.asarray(y, dtype=np.float64)

        x1 = x_a * self.cos_az - y_a * self.sin_az
        y1 = x_a * self.sin_az + y_a * self.cos_az

        sx = self.cx + self.scale * x1
        sy = self.cy - self.scale * (z_s * self.cos_el + y1 * self.sin_el)
        depth = z_s * self.sin_el - y1 * self.cos_el  # larger = closer to viewer
        return sx, sy, depth

    def project_pt(self, x: float, y: float, z: float):
        sx, sy, d = self.project(x, y, z)
        return float(sx), float(sy), float(d)


def sample_palette(t_vals, stops):
    """Interpolate RGB colors smoothly over normalized t in [0, 1]."""
    t = np.clip(t_vals, 0.0, 1.0)
    n = len(stops) - 1
    scaled = t * n
    idx = np.clip(np.floor(scaled).astype(int), 0, n - 1)
    frac = scaled - idx
    stops_arr = np.array(stops, dtype=np.float64)
    c0 = stops_arr[idx]
    c1 = stops_arr[idx + 1]
    return c0 + (c1 - c0) * frac[..., None]


DEFAULT_SURFACE_STOPS = [
    (22, 36, 78),     # Deep valley indigo
    (30, 82, 146),    # Mid slope cobalt
    (36, 152, 206),   # Upper cyan-teal
    (112, 214, 245),  # High ridge sky
    (248, 224, 118),  # Summit sunlit gold
]


def draw_3d_base_grid(draw: ImageDraw.ImageDraw, cam: Camera3D, xlim=(-2.5, 2.5), ylim=(-2.5, 2.5), z0=0.0, step=1.0, grid_color=(32, 46, 80)):
    """Draw reference coordinate grid on the horizontal plane z = z0."""
    xs = np.arange(xlim[0], xlim[1] + 1e-6, step)
    ys = np.arange(ylim[0], ylim[1] + 1e-6, step)
    for xv in xs:
        p0 = cam.project_pt(xv, ylim[0], z0)
        p1 = cam.project_pt(xv, ylim[1], z0)
        draw.line([p0[:2], p1[:2]], fill=grid_color, width=1)
    for yv in ys:
        p0 = cam.project_pt(xlim[0], yv, z0)
        p1 = cam.project_pt(xlim[1], yv, z0)
        draw.line([p0[:2], p1[:2]], fill=grid_color, width=1)


def draw_3d_axes(
    draw: ImageDraw.ImageDraw,
    cam: Camera3D,
    xlim=(-2.6, 2.8),
    ylim=(-2.6, 2.8),
    zlim=(0.0, 3.2),
    axis_color=(148, 163, 184),
    labels=("x", "y", "f(x, y)"),
):
    """Draw clean x, y, z 3D coordinate axes with arrowheads and labels."""
    font_lbl = get_font(22, "demibold")
    axes_defs = [
        ((xlim[0], 0, zlim[0]), (xlim[1], 0, zlim[0]), labels[0]),
        ((0, ylim[0], zlim[0]), (0, ylim[1], zlim[0]), labels[1]),
        ((xlim[0], ylim[0], zlim[0]), (xlim[0], ylim[0], zlim[1]), labels[2]),
    ]
    for p_start, p_end, lbl in axes_defs:
        a = cam.project_pt(*p_start)
        b = cam.project_pt(*p_end)
        draw.line([a[:2], b[:2]], fill=axis_color, width=2)
        dx, dy = b[0] - a[0], b[1] - a[1]
        L = math.hypot(dx, dy)
        if L > 12:
            ang = math.atan2(dy, dx)
            hl = 12
            pL = (b[0] - hl * math.cos(ang - 0.4), b[1] - hl * math.sin(ang - 0.4))
            pR = (b[0] - hl * math.cos(ang + 0.4), b[1] - hl * math.sin(ang + 0.4))
            draw.polygon([(b[0], b[1]), pL, pR], fill=axis_color)
            tx = b[0] + 22 * math.cos(ang)
            ty = b[1] + 22 * math.sin(ang)
            draw.text((tx, ty), lbl, font=font_lbl, fill=TEXT, anchor="mm")


def make_dashed_segments(p0, p1, n_dashes: int = 9):
    """Subdivide a 3D line segment p0->p1 into alternating dash pairs."""
    segs = []
    for k in range(n_dashes):
        u0 = k / n_dashes
        u1 = (k + 0.55) / n_dashes
        a = tuple(p0[i] + (p1[i] - p0[i]) * u0 for i in range(3))
        b = tuple(p0[i] + (p1[i] - p0[i]) * u1 for i in range(3))
        segs.append((a, b))
    return segs


def _draw_label_on_layer(layer: Image.Image, fd: ImageDraw.ImageDraw, x: float, y: float, label: str, col_rgba, is_tex: bool = False):
    if is_tex or ("$" in label):
        tex_img = render_tex_rgba(label, fontsize=18, color_rgb=col_rgba[:3], dpi=150)
        paste_rgba(layer, tex_img, int(x), int(y), alpha=col_rgba[3] / 255.0, anchor="center")
    else:
        font_lbl = get_font(20, "bold")
        fd.text((x, y), label, font=font_lbl, fill=col_rgba, anchor="mm")


_CONTOUR_CACHE = {}


def _get_cached_contours(X, Y, Z, contour_levels, valid_mask):
    key = (Z.shape, float(Z[0, 0]), float(Z[Z.shape[0] // 2, Z.shape[1] // 2]), tuple(float(v) for v in contour_levels))
    if key in _CONTOUR_CACHE:
        return _CONTOUR_CACHE[key]
    fig_c = plt.figure()
    Z_for_c = np.where(valid_mask, Z, np.nan) if valid_mask is not None else Z
    cs = plt.contour(X, Y, Z_for_c, levels=contour_levels)
    plt.close(fig_c)
    extracted = []
    for level_idx, segs in enumerate(cs.allsegs):
        z_lvl = float(contour_levels[level_idx]) + 0.02
        for seg in segs:
            if len(seg) >= 2:
                extracted.append((seg[:, 0].copy(), seg[:, 1].copy(), np.full(len(seg), z_lvl)))
    _CONTOUR_CACHE[key] = extracted
    return extracted


def render_surface_and_items(
    canvas: Image.Image,
    cam: Camera3D,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    z_range=None,
    palette=DEFAULT_SURFACE_STOPS,
    alpha: float = 0.94,
    wireframe_step: int = 4,
    contour_levels=None,
    contour_color=(167, 139, 250),
    valid_mask=None,
    extra_items=None,
):
    """Render shaded 3D surface quads, depth-sorted 3D contour curves, and foreground annotations."""
    if alpha <= 0.01:
        return

    sx, sy, depth = cam.project(X, Y, Z)
    ny, nx = Z.shape

    # Compute per-quad surface normals via diagonal cross product (works for Cartesian and polar meshes)
    P = np.stack([X, Y, Z * cam.z_scale], axis=-1)
    d1 = P[1:, 1:, :] - P[:-1, :-1, :]
    d2 = P[1:, :-1, :] - P[:-1, 1:, :]
    norms = np.cross(d1, d2)
    # Orient normals upward (+z)
    flip = np.where(norms[..., 2] < 0, -1.0, 1.0)[..., None]
    norms = norms * flip
    n_len = np.linalg.norm(norms, axis=-1, keepdims=True) + 1e-9
    norms = norms / n_len

    light = np.array([-0.42, -0.58, 0.70])
    light /= np.linalg.norm(light)
    ndotl = np.clip(np.sum(norms * light[None, None, :], axis=-1), 0.0, 1.0)
    q_shade = np.clip(0.46 + 0.62 * ndotl, 0.35, 1.08)

    zmin = float(np.nanmin(Z)) if z_range is None else z_range[0]
    zmax = float(np.nanmax(Z)) if z_range is None else z_range[1]
    t_norm = np.clip((Z - zmin) / max(1e-6, zmax - zmin), 0.0, 1.0)
    base_rgb = sample_palette(t_norm, palette)
    q_base = 0.25 * (base_rgb[:-1, :-1] + base_rgb[:-1, 1:] + base_rgb[1:, 1:] + base_rgb[1:, :-1])
    q_rgb = np.clip(q_base * q_shade[..., None], 0, 255).astype(np.int32)

    q_depth = 0.25 * (depth[:-1, :-1] + depth[:-1, 1:] + depth[1:, 1:] + depth[1:, :-1])

    if valid_mask is not None:
        q_valid = valid_mask[:-1, :-1] & valid_mask[:-1, 1:] & valid_mask[1:, 1:] & valid_mask[1:, :-1]
    else:
        q_valid = np.ones_like(q_depth, dtype=bool)

    # Build unified depth-sorted draw list of surface quads + contour segments
    draw_cmds = []
    sx00, sy00 = sx[:-1, :-1].ravel(), sy[:-1, :-1].ravel()
    sx01, sy01 = sx[:-1, 1:].ravel(), sy[:-1, 1:].ravel()
    sx11, sy11 = sx[1:, 1:].ravel(), sy[1:, 1:].ravel()
    sx10, sy10 = sx[1:, :-1].ravel(), sy[1:, :-1].ravel()
    r_flat = q_rgb[:, :, 0].ravel()
    g_flat = q_rgb[:, :, 1].ravel()
    b_flat = q_rgb[:, :, 2].ravel()
    d_flat = q_depth.ravel()
    v_flat = q_valid.ravel()

    cols_count = nx - 1
    a_fill = int(round(255 * clamp(alpha)))

    for idx in np.flatnonzero(v_flat):
        i_row = idx // cols_count
        i_col = idx % cols_count
        has_edge = bool(wireframe_step and (i_row % wireframe_step == 0 or i_col % wireframe_step == 0))
        draw_cmds.append((
            float(d_flat[idx]),
            0,  # quad
            idx,
            has_edge,
        ))

    contour_segs = []
    if contour_levels is not None and len(contour_levels) > 0:
        if contour_levels == "polar_rings":
            # Draw clean circular level curves along constant-r rows
            step_r = max(1, (ny - 1) // 5)
            for ir in range(step_r, ny, step_r):
                psx, psy, pd = cam.project(X[ir, :], Y[ir, :], Z[ir, :] + 0.02)
                for k in range(nx - 1):
                    seg_d = 0.5 * (float(pd[k]) + float(pd[k + 1])) + 0.14
                    c_idx = len(contour_segs)
                    contour_segs.append(((float(psx[k]), float(psy[k])), (float(psx[k + 1]), float(psy[k + 1]))))
                    draw_cmds.append((seg_d, 1, c_idx, False))
        else:
            for seg_x, seg_y, seg_z in _get_cached_contours(X, Y, Z, contour_levels, valid_mask):
                psx, psy, pd = cam.project(seg_x, seg_y, seg_z)
                for k in range(len(seg_x) - 1):
                    seg_d = 0.5 * (float(pd[k]) + float(pd[k + 1])) + 0.14
                    c_idx = len(contour_segs)
                    contour_segs.append(((float(psx[k]), float(psy[k])), (float(psx[k + 1]), float(psy[k + 1]))))
                    draw_cmds.append((seg_d, 1, c_idx, False))

    draw_cmds.sort(key=lambda item: (item[0], item[1]))

    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    c_rgba = (*contour_color[:3], int(round(235 * clamp(alpha))))

    for _, kind, idx, has_edge in draw_cmds:
        if kind == 0:
            poly = [
                (float(sx00[idx]), float(sy00[idx])),
                (float(sx01[idx]), float(sy01[idx])),
                (float(sx11[idx]), float(sy11[idx])),
                (float(sx10[idx]), float(sy10[idx])),
            ]
            col = (int(r_flat[idx]), int(g_flat[idx]), int(b_flat[idx]), a_fill)
            if has_edge:
                edge_col = (min(255, col[0] + 20), min(255, col[1] + 24), min(255, col[2] + 32), int(a_fill * 0.6))
                d.polygon(poly, fill=col, outline=edge_col)
            else:
                d.polygon(poly, fill=col)
        else:
            p_a, p_b = contour_segs[idx]
            d.line([p_a, p_b], fill=c_rgba, width=2)

    canvas_rgba = Image.alpha_composite(canvas.convert("RGBA"), layer)

    if extra_items:
        fg = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        fd = ImageDraw.Draw(fg)

        def item_depth(it):
            t = it["type"]
            if t == "point":
                return cam.project_pt(*it["p"])[2]
            if t == "arrow":
                return 0.5 * (cam.project_pt(*it["p0"])[2] + cam.project_pt(*it["p1"])[2])
            if t == "line":
                pts = it["pts"]
                return sum(cam.project_pt(*p)[2] for p in pts) / max(1, len(pts))
            return 0.0

        for it in sorted(extra_items, key=item_depth):
            t = it["type"]
            it_alpha = int(round(255 * clamp(it.get("alpha", 1.0))))
            if it_alpha <= 2:
                continue
            col = (*it["color"][:3], it_alpha)
            if t == "line":
                w = it.get("width", 3)
                if it.get("dashed", False) and len(it["pts"]) == 2:
                    for seg_a, seg_b in make_dashed_segments(it["pts"][0], it["pts"][1]):
                        pa = cam.project_pt(*seg_a)[:2]
                        pb = cam.project_pt(*seg_b)[:2]
                        fd.line([pa, pb], fill=col, width=w)
                else:
                    proj = [cam.project_pt(*p)[:2] for p in it["pts"]]
                    if len(proj) >= 2:
                        fd.line(proj, fill=col, width=w, joint="curve")
            elif t == "arrow":
                a = cam.project_pt(*it["p0"])
                b = cam.project_pt(*it["p1"])
                dx, dy = b[0] - a[0], b[1] - a[1]
                L = math.hypot(dx, dy)
                if L >= 3:
                    w = it.get("width", 5)
                    hl = min(20.0, L * 0.42)
                    ux, uy = dx / L, dy / L
                    bx, by = b[0] - ux * hl * 0.72, b[1] - uy * hl * 0.72
                    glow = (*it["color"][:3], int(it_alpha * 0.25))
                    fd.line([(a[0], a[1]), (bx, by)], fill=glow, width=w + 6)
                    fd.line([(a[0], a[1]), (bx, by)], fill=col, width=w)
                    ang = math.atan2(dy, dx)
                    pL = (b[0] - hl * math.cos(ang - 0.42), b[1] - hl * math.sin(ang - 0.42))
                    pR = (b[0] - hl * math.cos(ang + 0.42), b[1] - hl * math.sin(ang + 0.42))
                    fd.polygon([(b[0], b[1]), pL, pR], fill=col)
                    if it.get("label"):
                        off = it.get("label_offset", (36 * ux, 28 * uy - 12))
                        _draw_label_on_layer(fg, fd, b[0] + off[0], b[1] + off[1], it["label"], col, it.get("is_tex", False))
            elif t == "point":
                px, py, _ = cam.project_pt(*it["p"])
                r = it.get("radius", 9)
                if it.get("halo", True):
                    fd.ellipse([px - r * 2.1, py - r * 2.1, px + r * 2.1, py + r * 2.1], fill=(*it["color"][:3], int(it_alpha * 0.25)))
                fd.ellipse([px - r, py - r, px + r, py + r], fill=col, outline=(255, 255, 255, it_alpha), width=2)
                if it.get("label"):
                    off = it.get("label_offset", (58, -22))
                    _draw_label_on_layer(fg, fd, px + off[0], py + off[1], it["label"], col, it.get("is_tex", False))

        canvas_rgba = Image.alpha_composite(canvas_rgba, fg)

    canvas.paste(canvas_rgba.convert("RGB"))
