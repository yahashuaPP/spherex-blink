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
import astrometry  # noqa: E402
import science as sc  # noqa: E402
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


def bands(res) -> dict:
    """Comparación de épocas detector por detector (misma banda en todas las épocas).

    Cada exposición de SPHEREx ve una longitud de onda distinta; mezclar detectores hace
    que dos épocas difieran en brillo solo por la mezcla. Aquí cada banda se compara
    consigo misma. Solo salen las bandas presentes en al menos dos épocas.
    """
    out = {}
    for det in range(1, 7):
        have = [(e, e.get("by_detector", {}).get(det)) for e in res["epochs"]]
        have = [(e, im) for e, im in have if im is not None and np.isfinite(im).sum() > 20]
        if len(have) < 2:
            continue
        first, last = have[0][1], have[-1][1]
        norm = sc.compute_reference_norm(first)
        diff, _ = sc.compute_epoch_difference(first, last)
        ny, nx = first.shape
        centro = first[ny // 2 - 3:ny // 2 + 4, nx // 2 - 3:nx // 2 + 4]
        ruido = max(float(sc.mad_std(first, ignore_nan=True)), 1e-6)
        lo, hi = sc.DETECTOR_LVF_RANGES_UM[det]
        out[det] = {
            "label": f"Detector {det} ({lo:.2f}–{hi:.2f} µm)",
            "snr": float(np.nanmax(centro)) / ruido if np.isfinite(centro).any() else 0.0,
            "epochs": [{"date": Time(e["t_mean"], format="mjd").iso[:10],
                        "png": png(sc.array_to_png_base64(im, norm=norm))} for e, im in have],
            "diff": png(sc.array_to_png_base64(diff, divergent=True)),
        }
    return out


@app.get("/resolve")
def resolve(name: str):
    try:
        ra, dec = source.resolve_name(name)
    except Exception as err:
        raise HTTPException(404, f"No pude resolver '{name}': {err}")
    return {"name": name, "ra": ra, "dec": dec}


PRE = source.ROOT / "precomputed"


def pre_path(ra: float, dec: float, fov: float):
    return PRE / f"{ra:.4f}_{dec:.4f}_{fov}.json"


def compute(ra: float, dec: float, fov: float) -> dict:
    """Todo el cálculo para una posición: datos -> ciencia -> respuesta lista para la web."""
    exps = source.get_exposures(ra, dec, fov)
    if not exps:
        raise LookupError("SPHEREx no tiene exposiciones en esa posición.")
    res = build_frames(exps, ra, dec, fov)
    if not res["epochs"]:
        raise LookupError("Ninguna exposición cubre esa posición.")

    try:        # posición exposición por exposición (funciona también con estrellas saturadas)
        astro = astrometry.measure(exps, ra, dec, sc.match_known_target(ra, dec))
    except Exception as err:
        print("astrometry falló:", repr(err))
        astro = None
    saturated = bool(astro and astro["saturated"])
    if astro:
        astro["png"] = png(astro.pop("png_b64"))

    return clean({
        "ra": ra, "dec": dec, "fov": fov, "source": source.SOURCE, "n_exposures": len(exps),
        "epochs": [{
            "date": Time(e["t_mean"], format="mjd").iso[:10],
            "n_exp": e["n_exp"],
            "png": png(b),
        } for e, b in zip(res["epochs"], res["png_epochs_b64"])],
        "diff": png(res.get("png_diff_b64")),
        "sum": png(res.get("png_sum_b64")),
        "rgb": png(res.get("png_rgb_b64")),
        # Con el núcleo saturado la fotometría de apertura no es fiable: no se muestra.
        "spectrum": None if saturated else png(res.get("png_spectrum_b64")),
        "saturated": saturated,
        "astro": astro,
        "flags": png(res.get("png_flags_b64")),
        "detectors": {d: png(b) for d, b in (res.get("png_detectors_b64") or {}).items()},
        "bands": bands(res),
        "validation": res.get("validation"),
        "flags_summary": res.get("flags_summary"),
    })


@app.get("/frames")
def frames(ra: float, dec: float, fov: float = 0.1):
    key = (round(ra, 5), round(dec, 5), fov)
    if key not in _cache:
        pre = pre_path(ra, dec, fov)
        if pre.exists():                       # resultado ya preparado por pipeline.py: instantáneo
            import json
            _cache[key] = json.loads(pre.read_text())
        else:
            try:
                _cache[key] = compute(ra, dec, fov)
            except LookupError as err:
                raise HTTPException(404, str(err))
            except Exception as err:
                raise HTTPException(502, f"No pude obtener datos ({source.SOURCE}): {err}")
    return _cache[key]


@app.get("/")
def index():
    return FileResponse(source.ROOT / "web" / "index.html")
