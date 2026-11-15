"""API + servidor de la web.

Correr (parado en backend/):
    uvicorn api:app --reload
y abrir http://localhost:8000
"""
import matplotlib

matplotlib.use("Agg")  # sin ventanas: science.py dibuja PNGs dentro del servidor

import numpy as np  # noqa: E402
from astropy.time import Time  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402

import source  # noqa: E402  (agrega science/ al path)
from science import build_frames  # noqa: E402

app = FastAPI(title="SPHEREx Blink")
_cache: dict = {}


def png(b64):
    """science.py entrega el PNG en base64 'pelado'; el navegador necesita el prefijo."""
    return f"data:image/png;base64,{b64}" if b64 else None


def clean(obj):
    """Convierte tipos de numpy a tipos normales para poder mandarlos como JSON."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


@app.get("/resolve")
def resolve(name: str):
    try:
        ra, dec = source.resolve_name(name)
    except Exception as err:
        raise HTTPException(404, f"No pude resolver '{name}': {err}")
    return {"name": name, "ra": ra, "dec": dec}


@app.get("/frames")
def frames(ra: float, dec: float, fov: float = 0.1):
    key = (round(ra, 5), round(dec, 5), fov)
    if key not in _cache:
        try:
            exps = source.get_exposures(ra, dec, fov)
        except Exception as err:
            raise HTTPException(502, f"No pude obtener datos ({source.SOURCE}): {err}")
        if not exps:
            raise HTTPException(404, "SPHEREx no tiene exposiciones en esa posición.")

        res = build_frames(exps, ra, dec, fov)
        if not res["epochs"]:
            raise HTTPException(404, "Ninguna exposición cubre esa posición.")

        _cache[key] = clean({
            "ra": ra, "dec": dec, "fov": fov, "source": source.SOURCE,
            "epochs": [{
                "date": Time(e["t_mean"], format="mjd").iso[:10],
                "n_exp": e["n_exp"],
                "png": png(b),
            } for e, b in zip(res["epochs"], res["png_epochs_b64"])],
            "diff": png(res.get("png_diff_b64")),
            "sum": png(res.get("png_sum_b64")),
            "rgb": png(res.get("png_rgb_b64")),
            "spectrum": png(res.get("png_spectrum_b64")),
            "flags": png(res.get("png_flags_b64")),
            "detectors": {d: png(b) for d, b in (res.get("png_detectors_b64") or {}).items()},
            "validation": res.get("validation"),
            "flags_summary": res.get("flags_summary"),
        })
    return _cache[key]


@app.get("/")
def index():
    return FileResponse(source.ROOT / "web" / "index.html")
