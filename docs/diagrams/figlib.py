"""A very small SVG figure library for the RoomSense diagrams.

Style: the "Figure 1: overview" look of ML papers. Light background, rounded
pastel blocks colour-coded by role, lanes per machine, solid arrows for data
flow, dashed arrows for control signals, numbered steps, small logos and icons.

Icons are read from ./icons (vendored, see CREDITS.md) and embedded as data
URIs, so every SVG is a single self-contained file.
"""
import base64
import re
import subprocess
from pathlib import Path
from xml.sax.saxutils import escape

HERE = Path(__file__).resolve().parent
ICONS = HERE / "icons"
ASSETS = HERE / "assets"
FONT = "Inter, 'Helvetica Neue', Helvetica, Arial, sans-serif"
INK, MUTED, LINE = "#1F2933", "#5B6673", "#3D4756"
CONTROL = "#A23B72"  # dashed control / async signals

# role -> (fill, stroke, icon tint)
ROLES = {
    "data": ("#E3EEFB", "#6C9BD9", "#2F6DB5"),
    "train": ("#FFEBD6", "#E3A05B", "#B8691B"),
    "serve": ("#DFF3E6", "#62B283", "#2B7D4F"),
    "monitor": ("#ECE3F8", "#9B7FCF", "#6A4BA8"),
    "store": ("#FFF5CC", "#D4B446", "#8C7313"),
    "ext": ("#EEF1F4", "#9AA5B1", "#4B5563"),
}
ROLE_NAMES = {"data": "Data", "train": "Training", "serve": "Serving", "monitor": "Monitoring",
              "store": "Storage / registry", "ext": "People / external"}

try:  # exact text widths when Pillow and Inter are available, a good estimate otherwise
    from PIL import ImageFont
    _FONTS = {}

    def text_width(text: str, size: float, bold: bool = False) -> float:
        key = (round(size * 2), bold)
        if key not in _FONTS:
            name = "Inter-SemiBold.otf" if bold else "Inter-Regular.otf"
            _FONTS[key] = ImageFont.truetype(f"/usr/share/fonts/opentype/inter/{name}", key[0])
        return _FONTS[key].getlength(text) / 2
except (ImportError, OSError):
    def text_width(text: str, size: float, bold: bool = False) -> float:
        return len(text) * size * (0.6 if bold else 0.55)


def icon_uri(name: str, tint: str = "") -> str:
    """Data URI of icons/<name>.svg. Line icons (lu-*) are tinted via currentColor."""
    svg = (ICONS / f"{name}.svg").read_text()
    svg = re.sub(r"<!--.*?-->", "", svg, flags=re.S).strip()
    if tint:
        svg = svg.replace("currentColor", tint)
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


class Fig:
    def __init__(self, w: int, h: int, title: str = ""):
        self.w, self.h, self.title = w, h, title
        self.layers = {"lane": [], "edge": [], "node": [], "label": []}
        self.boxes = {}

    # --- primitives -----------------------------------------------------------
    def add(self, layer: str, svg: str) -> None:
        self.layers[layer].append(svg)

    def text(self, x, y, s, size=12, bold=False, fill=INK, anchor="start", layer="node", italic=False, mono=False):
        fam = "'DejaVu Sans Mono', Menlo, Consolas, monospace" if mono else FONT
        self.add(layer, f'<text x="{x:.1f}" y="{y:.1f}" font-family="{fam}" font-size="{size}" '
                        f'font-weight="{600 if bold else 400}" fill="{fill}" text-anchor="{anchor}"'
                        f'{" font-style=\"italic\"" if italic else ""}>{escape(str(s))}</text>')

    def icon(self, x, y, size, name, tint="", layer="node"):
        self.add(layer, f'<image x="{x:.1f}" y="{y:.1f}" width="{size}" height="{size}" href="{icon_uri(name, tint)}"/>')

    def image(self, x, y, w, h, path, radius=6, caption=""):
        """A small raster picture (camera frame, screenshot thumbnail), clipped to rounded corners."""
        p = Path(path)
        mime = "image/png" if p.suffix == ".png" else "image/jpeg"
        uri = f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode()
        cid = f"clip{len(self.layers['node'])}"
        self.add("node", f'<clipPath id="{cid}"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}"/></clipPath>'
                         f'<image x="{x}" y="{y}" width="{w}" height="{h}" preserveAspectRatio="xMidYMid slice" '
                         f'clip-path="url(#{cid})" href="{uri}"/>'
                         f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="none" stroke="#9AA5B1"/>')
        if caption:
            self.text(x + w / 2, y + h + 13, caption, 10.5, fill=MUTED, anchor="middle")

    def lane(self, x, y, w, h, label, icon=None, sub="", fill="#F7F9FB"):
        self.add("lane", f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{fill}" stroke="#C5CED8" '
                         f'stroke-width="1.2" stroke-dasharray="6 4"/>')
        tx = x + 14
        if icon:
            self.icon(x + 12, y + 9, 18, icon, MUTED, layer="lane")
            tx += 24
        self.text(tx, y + 23, label, 13.5, bold=True, fill="#374151", layer="lane")
        if sub:
            self.text(tx + text_width(label, 13.5, True) + 8, y + 23, sub, 11.5, fill=MUTED, layer="lane")

    def box(self, bid, x, y, w, h, title, sub=(), role="data", logo=None, icon=None, logos=(), dashed=False,
            title_size=13.5, mono_sub=False):
        """A rounded block. `logo` = full-colour brand logo, `icon` = line icon tinted in the role colour."""
        fill, stroke, tint = ROLES[role]
        self.boxes[bid] = (x, y, w, h)
        dash = ' stroke-dasharray="5 4"' if dashed else ""
        self.add("node", f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" stroke="{stroke}" '
                         f'stroke-width="1.4"{dash}/>')
        sub = [sub] if isinstance(sub, str) else list(sub)
        block_h = title_size + len(sub) * 14.5
        ty = y + (h - block_h) / 2 + title_size - 1
        tx = x + 12
        mark = logo or icon
        if mark:
            size = 24 if h >= 50 else 18
            if logo:
                self.icon(x + 10, y + (h - size) / 2, size, logo)
            else:
                self.icon(x + 10, y + (h - size) / 2, size, icon, tint)
            tx = x + 10 + size + 8
        self.text(tx, ty, title, title_size, bold=True)
        for i, line in enumerate(sub):
            self.text(tx, ty + 15 + i * 14.5, line, 11 if not mono_sub else 10.5, fill=MUTED, mono=mono_sub)
        lx = x + w - 8
        for name in reversed(list(logos)):  # extra small logos, top-right corner
            lx -= 16
            self.icon(lx, y + 6, 14, name)
            lx -= 3

    def chip(self, x, y, label, role="store", icon=None, size=10.5, mono=True):
        """A small pill for an artifact or a file name. Returns its width."""
        fill, stroke, tint = ROLES[role]
        w = (len(label) * size * 0.615 if mono else text_width(label, size)) + 18 + (14 if icon else 0)
        self.add("node", f'<rect x="{x}" y="{y}" width="{w:.1f}" height="20" rx="10" fill="#fff" stroke="{stroke}"/>')
        tx = x + 8
        if icon:
            self.icon(x + 6, y + 4, 12, icon, tint)
            tx += 14
        self.text(tx, y + 14, label, size, fill="#374151", mono=mono)
        return w

    def step(self, x, y, n, color=LINE):
        self.add("label", f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9" fill="{color}"/>')
        self.text(x, y + 4, n, 11, bold=True, fill="#fff", anchor="middle", layer="label")

    # --- arrows ---------------------------------------------------------------
    def anchor(self, bid, side, shift=0.0):
        x, y, w, h = self.boxes[bid]
        return {"l": (x, y + h / 2 + shift), "r": (x + w, y + h / 2 + shift),
                "t": (x + w / 2 + shift, y), "b": (x + w / 2 + shift, y + h)}[side]

    def arrow(self, pts, label="", dashed=False, step=None, color=None, both=False, lpos=0.5, ldx=0, ldy=0,
              width=1.6, label_bg=True, step_at=None):
        """Polyline arrow with rounded corners through `pts` [(x, y), ...]."""
        color = color or (CONTROL if dashed else LINE)
        d = f"M{pts[0][0]:.1f},{pts[0][1]:.1f}"
        r = 7
        for i in range(1, len(pts)):
            x, y = pts[i]
            if i < len(pts) - 1:  # round the corner at pts[i]
                px, py = pts[i - 1]
                nx, ny = pts[i + 1]
                d1 = max(abs(x - px) + abs(y - py), 1e-6)
                d2 = max(abs(nx - x) + abs(ny - y), 1e-6)
                rr = min(r, d1 / 2, d2 / 2)
                ax, ay = x - (x - px) / d1 * rr, y - (y - py) / d1 * rr
                bx, by = x + (nx - x) / d2 * rr, y + (ny - y) / d2 * rr
                d += f" L{ax:.1f},{ay:.1f} Q{x:.1f},{y:.1f} {bx:.1f},{by:.1f}"
            else:
                d += f" L{x:.1f},{y:.1f}"
        marker = "ctl" if color == CONTROL else "ink"
        attrs = f'marker-end="url(#arrow-{marker})"' + (f' marker-start="url(#arrow-{marker}-rev)"' if both else "")
        dash = ' stroke-dasharray="6 4"' if dashed else ""
        self.add("edge", f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}"{dash} {attrs} '
                         f'stroke-linejoin="round"/>')
        # label at fraction `lpos` of the total length
        seg = [((pts[i][0], pts[i][1]), (pts[i + 1][0], pts[i + 1][1])) for i in range(len(pts) - 1)]
        lens = [abs(b[0] - a[0]) + abs(b[1] - a[1]) for a, b in seg]

        def at(frac):
            t = frac * sum(lens)
            for (a, b), ln in zip(seg, lens):
                if t <= ln or (a, b) == seg[-1]:
                    f = t / ln if ln else 0
                    return a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f
                t -= ln
            return pts[-1]

        lx, ly = at(lpos)
        lx, ly = lx + ldx, ly + ldy
        if label:
            lines = label.split("\n")
            w = max(text_width(s, 10.5) for s in lines) + 10
            h = 14 * len(lines) + 2
            if label_bg:
                self.add("label", f'<rect x="{lx - w / 2:.1f}" y="{ly - h / 2:.1f}" width="{w:.1f}" height="{h}" rx="4" '
                                  f'fill="#fff" fill-opacity="0.92"/>')
            for i, s in enumerate(lines):
                self.text(lx, ly - h / 2 + 12 + i * 14, s, 10.5, fill=color if dashed else "#374151", anchor="middle",
                          layer="label")
        if step is not None:
            sx, sy = at(step_at) if step_at is not None else (lx - (text_width(label.split("\n")[0], 10.5) / 2 + 14
                                                                    if label else 0), ly)
            self.step(sx, sy, step, color)

    def link(self, a, b, sa="r", sb="l", shift_a=0.0, shift_b=0.0, via=None, **kw):
        """Arrow between two boxes with an automatic elbow (or explicit `via` points)."""
        p1, p2 = self.anchor(a, sa, shift_a), self.anchor(b, sb, shift_b)
        if via is None:
            if sa in "lr" and sb in "lr":
                mx = (p1[0] + p2[0]) / 2
                via = [] if abs(p1[1] - p2[1]) < 1 else [(mx, p1[1]), (mx, p2[1])]
            elif sa in "tb" and sb in "tb":
                my = (p1[1] + p2[1]) / 2
                via = [] if abs(p1[0] - p2[0]) < 1 else [(p1[0], my), (p2[0], my)]
            elif sa in "lr":
                via = [(p2[0], p1[1])]
            else:
                via = [(p1[0], p2[1])]
        self.arrow([p1, *via, p2], **kw)

    # --- legend + output --------------------------------------------------------
    def legend(self, x, y, roles=("data", "train", "serve", "monitor", "store", "ext"), lines=True, steps=False):
        cx = x
        for r in roles:
            fill, stroke, _ = ROLES[r]
            self.add("label", f'<rect x="{cx}" y="{y - 10}" width="16" height="12" rx="3" fill="{fill}" stroke="{stroke}"/>')
            self.text(cx + 21, y, ROLE_NAMES[r], 11, fill="#374151", layer="label")
            cx += 21 + text_width(ROLE_NAMES[r], 11) + 16
        if lines:
            self.add("label", f'<path d="M{cx},{y - 4} h26" stroke="{LINE}" stroke-width="1.6" marker-end="url(#arrow-ink)"/>')
            self.text(cx + 33, y, "data flow", 11, fill="#374151", layer="label")
            cx += 33 + text_width("data flow", 11) + 16
            self.add("label", f'<path d="M{cx},{y - 4} h26" stroke="{CONTROL}" stroke-width="1.6" stroke-dasharray="6 4" '
                              f'marker-end="url(#arrow-ctl)"/>')
            self.text(cx + 33, y, "control / async signal", 11, fill="#374151", layer="label")
            cx += 33 + text_width("control / async signal", 11) + 16
        if steps:
            self.step(cx + 9, y - 4, 1)
            self.text(cx + 23, y, "order of events", 11, fill="#374151", layer="label")

    def svg(self) -> str:
        def marker(mid, color, rev=False):
            path = "M10,0 L0,5 L10,10 z" if rev else "M0,0 L10,5 L0,10 z"
            return (f'<marker id="{mid}" viewBox="0 0 10 10" refX="{1 if rev else 9}" refY="5" markerWidth="7" '
                    f'markerHeight="7" orient="auto"><path d="{path}" fill="{color}"/></marker>')
        defs = "".join([marker("arrow-ink", LINE), marker("arrow-ctl", CONTROL),
                        marker("arrow-ink-rev", LINE, True), marker("arrow-ctl-rev", CONTROL, True)])
        body = "\n".join("\n".join(self.layers[k]) for k in ("lane", "edge", "node", "label"))
        title = f"<title>{escape(self.title)}</title>" if self.title else ""
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" '
                f'height="{self.h}" font-family="{FONT}">{title}<defs>{defs}</defs>'
                f'<rect width="{self.w}" height="{self.h}" fill="#FFFFFF"/>\n{body}\n</svg>\n')

    def save(self, name: str, out: Path = HERE) -> Path:
        """Write <name>.svg, then PNG (2x) and PDF with rsvg-convert when it is installed."""
        svg = out / f"{name}.svg"
        svg.write_text(self.svg())
        for fmt, extra in (("png", ["-z", "2"]), ("pdf", [])):
            try:
                subprocess.run(["rsvg-convert", "-f", fmt, *extra, str(svg), "-o", str(out / f"{name}.{fmt}")],
                               check=True, capture_output=True)
            except (OSError, subprocess.CalledProcessError) as e:
                print(f"  ({fmt} not written: {e}; install librsvg2-bin)")
        return svg
