"""science/astrometry.py -- Posición de la fuente central exposición por exposición.

Por qué existe: con datos REALES, las estrellas brillantes (Barnard) saturan el detector.
El núcleo queda inservible y un centroide normal falla. Aquí se ajusta un perfil simétrico
(Moffat) a las ALAS de la estrella en cada exposición por separado, excluyendo los píxeles
saturados y sus vecinos. Cada exposición da una posición en el cielo; por época se reporta
la mediana y su error estadístico (dispersión / raíz de N).

Probado con 22 exposiciones reales de Barnard (sep-2025 y mar-2026): dispersión de
0.3-0.5" por exposición (1/15 de píxel).

Límite conocido: el detector gira 180° entre visitas separadas seis meses. Si las alas de una
estrella saturada no son simétricas, eso mete un error sistemático que cambia de signo entre
esas visitas (en Barnard: ~1.9" de exceso en RA). Entre visitas separadas un año la
orientación es la misma y ese error se cancela.
"""
from __future__ import annotations

import base64
import io
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import get_body_barycentric
from astropy.time import Time
from scipy.ndimage import binary_dilation
from scipy.optimize import least_squares

OVERFLOW_BIT = 1
# Píxeles que no se usan para medir posición: saturados (1) y defectuosos.
HARD_BAD = sum(1 << b for b in (1, 6, 9, 10, 11, 14, 15, 17, 19))
PIX_ARCSEC = 6.15


def fit_position(exp: Dict[str, Any], ra: float, dec: float, radius_px: float = 8.0):
    """Ajusta un Moffat + fondo a la fuente más cercana a (ra, dec) en UNA exposición.

    Devuelve (ra_medida, dec_medida, saturada) o None si no hay señal suficiente.
    """
    im = np.asarray(exp["image"], dtype=float)
    var = np.asarray(exp["variance"], dtype=float)
    fl = np.asarray(exp["flags"]).astype(np.uint32)
    ny, nx = im.shape
    x0, y0 = (float(v) for v in exp["wcs"].all_world2pix(ra, dec, 0, quiet=True))
    if not (3 <= x0 < nx - 3 and 3 <= y0 < ny - 3):
        return None
    xg, yg = x0, y0
    yy, xx = np.mgrid[0:ny, 0:nx]
    sat = (((fl >> OVERFLOW_BIT) & 1) == 1) | ~np.isfinite(im)
    saturada = bool(sat[int(round(y0)) - 2:int(round(y0)) + 3, int(round(x0)) - 2:int(round(x0)) + 3].any())
    usable = np.isfinite(im) & np.isfinite(var) & (var > 0) & ((fl & HARD_BAD) == 0) & ~binary_dilation(sat)

    sol = None
    for _ in range(3):
        ok = usable & (np.hypot(xx - x0, yy - y0) < radius_px)
        if ok.sum() < 25:
            return None
        X, Y, Z = xx[ok], yy[ok], im[ok]
        S = np.sqrt(var[ok] + (0.05 * np.abs(Z)) ** 2)      # 5 % extra: el modelo no es perfecto

        def resid(p):
            cx, cy, amp, a, b, bkg = p
            return (amp * (1 + ((X - cx) ** 2 + (Y - cy) ** 2) / a**2) ** (-b) + bkg - Z) / S

        try:
            sol = least_squares(
                resid, [x0, y0, max(np.nanmax(Z), 1e-3) * 3, 1.5, 1.5, float(np.nanmedian(Z))],
                loss="soft_l1",
                bounds=([xg - 3, yg - 3, 0, 0.2, 0.5, -1e4], [xg + 3, yg + 3, 1e10, 10, 6, 1e5]))
        except Exception:
            return None
        x0, y0 = sol.x[0], sol.x[1]

    cx, cy, amp, a, b, bkg = sol.x
    ruido = float(np.nanmedian(np.sqrt(var[usable]))) if usable.any() else np.inf
    if amp < 5 * ruido or abs(cx - xg) > 2.9 or abs(cy - yg) > 2.9:
        return None                                          # sin fuente clara, o ajuste pegado al borde
    ra_m, dec_m = (float(v) for v in exp["wcs"].all_pix2world(cx, cy, 0))
    return ra_m, dec_m, saturada


def parallax_offset(ra: float, dec: float, parallax_arcsec: float, mjd: float):
    """Desplazamiento aparente por paralaje (RA*cos(dec), Dec) en segundos de arco."""
    X, Y, Z = get_body_barycentric("earth", Time(mjd, format="mjd")).xyz.to_value("au")
    a, d = np.radians(ra), np.radians(dec)
    return (parallax_arcsec * (X * np.sin(a) - Y * np.cos(a)),
            parallax_arcsec * (X * np.cos(a) * np.sin(d) + Y * np.sin(a) * np.sin(d) - Z * np.cos(d)))


def measure(exposures: List[Dict[str, Any]], ra: float, dec: float,
            target: Optional[Dict[str, Any]] = None, gap_days: float = 30.0) -> Optional[Dict[str, Any]]:
    """Posición por exposición -> posición por época -> desplazamiento entre épocas.

    `target`: entrada de catálogo (pm_ra_cosdec_mas_yr, pm_dec_mas_yr, parallax_mas) si se conoce;
    con ella se calcula el desplazamiento esperado (movimiento propio + paralaje).
    """
    pts = []
    for e in exposures:
        r = fit_position(e, ra, dec)
        if r:
            pts.append((float(e["t"]), (r[0] - ra) * np.cos(np.radians(dec)) * 3600,
                        (r[1] - dec) * 3600, r[2], int(e.get("detector", 0))))
    if len(pts) < 3:
        return None
    pts.sort()
    P = np.array([p[:3] for p in pts])
    sat = np.array([p[3] for p in pts])

    grupos, ini = [], 0
    for i in range(1, len(P) + 1):
        if i == len(P) or P[i, 0] - P[i - 1, 0] > gap_days:
            grupos.append(slice(ini, i))
            ini = i

    epochs = []
    for g in grupos:
        t, x, y = P[g, 0], P[g, 1], P[g, 2]
        n = len(t)
        if n < 3:
            continue
        epochs.append({
            "date": Time(float(t.mean()), format="mjd").iso[:10], "t": float(t.mean()), "n": n,
            "dra": float(np.median(x)), "ddec": float(np.median(y)),
            "dra_err": float(x.std(ddof=1) / np.sqrt(n)), "ddec_err": float(y.std(ddof=1) / np.sqrt(n)),
            "slice": g,
        })
    out: Dict[str, Any] = {"n_exposures": len(P), "saturated": bool(sat.mean() > 0.5), "epochs": epochs,
                           "shift": None, "expected": None}

    if len(epochs) >= 2:
        a, b = epochs[0], epochs[-1]
        dt = (b["t"] - a["t"]) / 365.25
        sh = {"from": a["date"], "to": b["date"], "dt_yr": dt,
              "ra": b["dra"] - a["dra"], "dec": b["ddec"] - a["ddec"],
              "ra_err": float(np.hypot(a["dra_err"], b["dra_err"])),
              "dec_err": float(np.hypot(a["ddec_err"], b["ddec_err"]))}
        sh["total"] = float(np.hypot(sh["ra"], sh["dec"]))
        # ¿misma orientación del detector? (visitas separadas ~1 año) -> el sistemático se cancela
        sh["same_orientation"] = bool(abs((dt * 2) - round(dt * 2)) < 0.15 and round(dt * 2) % 2 == 0)
        out["shift"] = sh
        if target and (abs(target.get("pm_ra_cosdec_mas_yr", 0)) + abs(target.get("pm_dec_mas_yr", 0))) > 0:
            pm_ra = target["pm_ra_cosdec_mas_yr"] / 1000 * dt
            pm_dec = target["pm_dec_mas_yr"] / 1000 * dt
            plx = target.get("parallax_mas", 0) / 1000
            pa = np.mean([parallax_offset(ra, dec, plx, t) for t in P[a["slice"], 0]], axis=0)
            pb = np.mean([parallax_offset(ra, dec, plx, t) for t in P[b["slice"], 0]], axis=0)
            out["expected"] = {"pm_ra": pm_ra, "pm_dec": pm_dec,
                               "par_ra": float(pb[0] - pa[0]), "par_dec": float(pb[1] - pa[1]),
                               "ra": float(pm_ra + pb[0] - pa[0]), "dec": float(pm_dec + pb[1] - pa[1])}
            out["expected"]["total"] = float(np.hypot(out["expected"]["ra"], out["expected"]["dec"]))

    out["png_b64"] = _plot(P, epochs, out, target)
    for ep in epochs:
        ep.pop("slice", None)
        ep.pop("t", None)
    return out


def _plot(P, epochs, out, target) -> str:
    colores = ["#00e1ff", "#ff9f1c", "#b388ff", "#06d6a0", "#ef476f", "#ffd166"]
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(5.6, 5.0), dpi=130, facecolor="#070a10")
    ax.set_facecolor("#070a10")
    for i, ep in enumerate(epochs):
        g, c = ep["slice"], colores[i % len(colores)]
        ax.scatter(P[g, 1], P[g, 2], s=16, color=c, alpha=0.55, linewidths=0)
        ax.errorbar(ep["dra"], ep["ddec"], xerr=ep["dra_err"], yerr=ep["ddec_err"], fmt="o", ms=8,
                    color=c, mec="white", mew=1.0, ecolor="white", capsize=3,
                    label=f"{ep['date']} (n={ep['n']})")
    if out.get("expected") and epochs:
        a, ex = epochs[0], out["expected"]
        ax.annotate("", xy=(a["dra"] + ex["ra"], a["ddec"] + ex["dec"]), xytext=(a["dra"], a["ddec"]),
                    arrowprops=dict(arrowstyle="->", color="#ffffff", lw=1.6, ls="--"))
        ax.plot([], [], "--", color="#ffffff", label="expected (proper motion + parallax)")
    ax.invert_xaxis()                                   # Este a la izquierda, como en el cielo
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel('ΔRA·cos(Dec) ["]   (East ←)', color="#c9d1d9")
    ax.set_ylabel('ΔDec ["]   (North ↑)', color="#c9d1d9")
    nombre = (target or {}).get("name", "Central source")
    ax.set_title(f"{nombre}: position in each exposure\n(1 SPHEREx pixel = {PIX_ARCSEC}\")",
                 fontsize=10, weight="bold", color="#ffffff")
    ax.grid(color="#21262d", ls="--", lw=0.7)
    ax.legend(fontsize=7.5, facecolor="#0d1117", edgecolor="#30363d", loc="best")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="PNG", facecolor="#070a10")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")
