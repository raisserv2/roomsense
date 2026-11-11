# Diagram credits

All figures in this folder are drawn by `make_diagrams.py` (our own code, `figlib.py`). The small logos and icons inside them come from three open-licensed icon sets, copied into `icons/` so the figures can be rebuilt offline.

| Icons | Source | Licence |
|---|---|---|
| Docker, PyTorch, FastAPI, Grafana, Prometheus, Git, GitHub, nginx, Python, SQLite, Ubuntu, pytest (full-colour logos) | [devicon](https://devicon.dev) npm package `devicon` | MIT |
| DVC, MLflow, Ray, Tailscale, ONNX, Telegram, NVIDIA, Lenovo, Uptime Kuma (single-colour logos, filled with each brand's official colour) | [Simple Icons](https://simpleicons.org) npm package `simple-icons` | CC0 1.0 |
| Concept icons, files named `lu-*.svg` (camera, phone, server, GPU, database, bell, gauge, user and so on) | [Lucide](https://lucide.dev) npm package `lucide-static` | ISC |

The logos are trademarks of their owners and are used only to name the tool a block stands for. `assets/sample-frame.svg` (the drawn room) is our own illustration; it stands in for a real camera frame so no private picture is published.

## Rebuilding

```bash
sudo apt install librsvg2-bin          # rsvg-convert, for PNG and PDF
python docs/diagrams/make_diagrams.py  # writes SVG + PNG + PDF for all seven figures
```

To use real pictures, put `frame.jpg` (a camera frame) in `assets/` and rerun.
