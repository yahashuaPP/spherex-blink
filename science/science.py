"""Ciencia de las imágenes — VERSIÓN BASE para que la arquitectura corra de punta a punta.
Ezequiel: este archivo es tuyo; mejora lo que está marcado con TODO.

Contrato:
    build_frames(exposures, ra, dec, fov) -> {"epochs", "diff", "spectrum", "wcs"}
"""
from collections import defaultdict

import numpy as np
from astropy.wcs import WCS
from reproject import reproject_interp

PIX_DEG = 6.15 / 3600   # píxel de SPHEREx en grados

# Bits de FLAGS que invalidan un píxel.
# TODO (Ezequiel): con datos reales, ajustar según la Tabla 8 del Explanatory
# Supplement. OJO: no enmascarar el bit que marca "aquí hay una fuente",
# o se borra justo la estrella que queremos ver.
BAD_BITS = 0b1


def common_wcs(ra: float, dec: float, fov: float):
    """Grilla común: norte arriba, este a la izquierda, centrada en (ra, dec)."""
    n = max(int(round(fov / PIX_DEG)), 5)
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [ra, dec]
    w.wcs.crpix = [(n + 1) / 2, (n + 1) / 2]
    w.wcs.cdelt = [-PIX_DEG, PIX_DEG]
    return w, (n, n)


def coadd(exposures: list[dict], wcs: WCS, shape) -> np.ndarray:
    """Media pesada por 1/varianza sobre la grilla común."""
    num = np.zeros(shape)
    den = np.zeros(shape)
    for e in exposures:
        img = e["image"].astype(float)
        var = e["variance"].astype(float)
        bad = ((e["flags"] & BAD_BITS) != 0) | ~np.isfinite(img) | ~(var > 0)
        img[bad] = np.nan
        peso = np.where(bad, 0.0, 1.0 / np.where(var > 0, var, 1.0))
        r, _ = reproject_interp((img, e["wcs"]), wcs, shape_out=shape)
        rw, _ = reproject_interp((peso, e["wcs"]), wcs, shape_out=shape)
        ok = np.isfinite(r) & np.isfinite(rw) & (rw > 0)
        num[ok] += r[ok] * rw[ok]
        den[ok] += rw[ok]
    out = np.full(shape, np.nan)
    out[den > 0] = num[den > 0] / den[den > 0]
    # TODO (Ezequiel): restar el fondo zodiacal (HDU ZODI) antes de sumar.
    return out - np.nanmedian(out)


def build_frames(exposures: list[dict], ra: float, dec: float, fov: float) -> dict:
    wcs, shape = common_wcs(ra, dec, fov)

    grupos = defaultdict(list)
    for e in exposures:
        grupos[e["epoch"]].append(e)

    epochs = []
    for ep in sorted(grupos):
        g = grupos[ep]
        epochs.append({
            "t_mean": float(np.mean([e["t"] for e in g])),
            "image": coadd(g, wcs, shape),
            "n_exp": len(g),
        })

    diff = epochs[-1]["image"] - epochs[0]["image"] if len(epochs) >= 2 else None
    # TODO (Ezequiel): espectro con fotometría de apertura + WCS-WAVE.
    return {"epochs": epochs, "diff": diff, "spectrum": None, "wcs": wcs}
