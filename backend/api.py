"""API + servidor de la web.

Correr (parado en backend/):
    uvicorn api:app --reload
y abrir http://localhost:8000
"""
import base64
import io
import sys
from pathlib import Path

import numpy as np
from astropy.time import Time
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "science"))

from science import build_frames  # noqa: E402
import source  # noqa: E402

app = FastAPI(title="SPHEREx Blink")
_cache: dict = {}


def to_png(img: np.ndarray, lo: float, hi: float, size: int = 420) -> str:
    """Matriz -> PNG en base64 (escala de grises, estirado asinh)."""
    x = np.nan_to_num((img - lo) / max(hi - lo, 1e-9), nan=0.0)
    x = np.clip(np.arcsinh(10 * np.clip(x, 0, None)) / np.arcsinh(10), 0, 1)
    im = Image.fromarray((255 * x[::-1]).astype("uint8"))   # [::-1]: norte arriba
    im = im.resize((size, size), Image.NEAREST)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def diff_png(img: np.ndarray, size: int = 420) -> str:
    """Diferencia: gris = sin cambio, blanco = apareció, negro = desapareció."""
    s = np.nanpercentile(np.abs(img), 99.5) or 1.0
    x = np.clip(np.nan_to_num(img / s, nan=0.0), -1, 1) * 0.5 + 0.5
    im = Image.fromarray((255 * x[::-1]).astype("uint8")).resize((size, size), Image.NEAREST)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@app.get("/resolve")
def resolve(name: str):
    try:
        ra, dec = source.resolve_name(name)
    except Exception as err:
        raise HTTPException(404, f"No pude resolver '{name}': {err}")
    return {"name": name, "ra": ra, "dec": dec}


@app.get("/frames")
def frames(ra: float, dec: float, fov: float = 0.05):
    key = (round(ra, 5), round(dec, 5), fov)
    if key not in _cache:
        try:
            exps = source.get_exposures(ra, dec, fov)
        except Exception as err:
            raise HTTPException(502, f"No pude obtener datos ({source.SOURCE}): {err}")
        if not exps:
            raise HTTPException(404, "SPHEREx no tiene exposiciones en esa posición.")
        res = build_frames(exps, ra, dec, fov)
        todas = np.concatenate([e["image"].ravel() for e in res["epochs"]])
        lo, hi = np.nanpercentile(todas, [5, 99.8])        # misma escala en todas las épocas
        _cache[key] = {
            "ra": ra, "dec": dec, "fov": fov, "source": source.SOURCE,
            "epochs": [{
                "date": Time(e["t_mean"], format="mjd").iso[:10],
                "n_exp": e["n_exp"],
                "png": to_png(e["image"], lo, hi),
            } for e in res["epochs"]],
            "diff": diff_png(res["diff"]) if res["diff"] is not None else None,
        }
    return _cache[key]


@app.get("/")
def index():
    return FileResponse(ROOT / "web" / "index.html")
