"""Y1 — Buscar en IRSA las exposiciones de SPHEREx que cubren (ra, dec).

Uso rápido:
    python query.py "Barnard's Star"
    python query.py 269.446 4.768
"""
import re
import sys

import pyvo
from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as u

TAP_URL = "https://irsa.ipac.caltech.edu/TAP"
EPOCH_GAP_DAYS = 30.0  # salto mayor a esto = época nueva


# Objetos de prueba del reto, para no depender de SIMBAD en la demo.
# nombre normalizado: (ra, dec en grados, pmra*cos(dec), pmdec en mas/año, época de la posición)
# OJO: valores puestos de memoria; verificarlos contra SIMBAD/Gaia cuando vuelva el servicio.
KNOWN = {
    "barnardsstar": (269.45208, 4.69336, -802.8, 10362.5, 2000.0),
    "epsilonindib": (331.04383, -56.78269, 3955.0, -2466.0, 2000.0),
    "wise0855": (133.79513, -7.24515, -8118.0, 680.0, 2014.3),  # coords del reto; época por confirmar
    "rwaurigae": (76.9564, 30.4014, 0.0, 0.0, 2000.0),
    "rdraconis": (248.1675, 66.7550, 0.0, 0.0, 2000.0),
}


def resolve(name: str, obstime: str = "2025-11-01") -> tuple[float, float]:
    """Nombre -> (ra, dec) en grados, corregido por movimiento propio a `obstime`.

    SIMBAD da la posición en J2000; Barnard se movió ~4' desde entonces.
    """
    key = re.sub(r"[^a-z0-9]", "", name.lower())
    if key in KNOWN:
        ra0, dec0, pmra, pmdec, epoch = KNOWN[key]
    else:
        try:  # SIMBAD: trae movimiento propio
            from astroquery.simbad import Simbad

            s = Simbad()
            s.add_votable_fields("pmra", "pmdec")
            row = s.query_object(name)[0]
            ra0, dec0 = float(row["ra"]), float(row["dec"])
            pmra, pmdec, epoch = float(row["pmra"] or 0), float(row["pmdec"] or 0), 2000.0
        except Exception as err:  # SIMBAD caído: Sesame (sin movimiento propio)
            print(f"SIMBAD no respondió ({type(err).__name__}); uso Sesame sin movimiento propio")
            c = SkyCoord.from_name(name)
            return float(c.ra.deg), float(c.dec.deg)

    c = SkyCoord(
        ra=ra0 * u.deg,
        dec=dec0 * u.deg,
        pm_ra_cosdec=pmra * u.mas / u.yr,
        pm_dec=pmdec * u.mas / u.yr,
        obstime=Time(epoch, format="jyear"),
    )
    c = c.apply_space_motion(new_obstime=Time(obstime))
    return float(c.ra.deg), float(c.dec.deg)


def find_exposures(ra: float, dec: float, fov_deg: float = 0.05) -> list[dict]:
    """Lista de exposiciones ordenadas por tiempo.

    Cada elemento: {"url", "t" (MJD), "detector" (1..6), "epoch" (0, 1, 2...)}
    La url ya incluye el recorte (?center=..&size=..).
    """
    local = _from_index(ra, dec, fov_deg)
    if local:
        print(f"Usando el índice local ({len(local)} exposiciones); no hace falta el buscador de IRSA.")
        assign_epochs(local)
        return local

    tap = pyvo.dal.TAPService(TAP_URL)
    query = f"""
    SELECT 'https://irsa.ipac.caltech.edu/' || a.uri AS uri,
           p.time_bounds_lower AS t
    FROM spherex.artifact a
    JOIN spherex.plane p ON a.planeid = p.planeid
    WHERE 1 = CONTAINS(POINT('ICRS', {ra}, {dec}), p.poly)
    ORDER BY p.time_bounds_lower
    """
    table = _run_query(tap, query, ra, dec)

    out = []
    for row in table:
        uri = str(row["uri"])
        if not uri.startswith("http"):
            uri = "https://irsa.ipac.caltech.edu/" + uri.lstrip("/")
        m = re.search(r"D(\d)_spx", uri)  # ..._2D4_spx_... -> detector 4
        out.append({
            "url": f"{uri}?center={ra},{dec}d&size={fov_deg}",
            "t": float(row["t"]),
            "detector": int(m.group(1)) if m else 0,
        })
    assign_epochs(out)
    return out


def _from_index(ra, dec, fov_deg, tol_deg=0.05):
    """Exposiciones desde nuestros índices propios (no usan el buscador de IRSA):
    - backend/index/<objeto>.json  -> lo que genera s3_index.py
    - backend/index/sky/           -> el índice de todo el cielo (sky_index.py)
    """
    import json
    from pathlib import Path

    files = {}
    here = SkyCoord(ra * u.deg, dec * u.deg)
    for f in sorted((Path(__file__).resolve().parent / "index").glob("*.json")):
        if f.name.startswith(("_", "pos_")):
            continue
        d = json.loads(f.read_text())
        if here.separation(SkyCoord(d["ra"] * u.deg, d["dec"] * u.deg)).deg < tol_deg:
            files.update({x["key"]: x for x in d["files"]})
    try:
        import sky_index
        files.update({x["key"]: x for x in sky_index.find(ra, dec)})
    except Exception as err:
        print(f"  (índice de todo el cielo no disponible: {type(err).__name__}: {err})")
    if not files:
        return None
    return sorted(
        ({"url": f"https://nasa-irsa-spherex.s3.amazonaws.com/{x['key']}#{ra},{dec},{fov_deg}",
          "key": x["key"], "t": x["t"], "detector": x["detector"]} for x in files.values()),
        key=lambda e: e["t"])


def _from_index_offline(ra, dec, tol_deg=0.05):
    """Lo que ya se sabe de una posición SIN usar la red (para el reporte de estado)."""
    import json
    from pathlib import Path

    idx = Path(__file__).resolve().parent / "index"
    files = {}
    here = SkyCoord(ra * u.deg, dec * u.deg)
    for f in sorted(idx.glob("*.json")):
        if f.name.startswith("_"):
            continue
        d = json.loads(f.read_text())
        if here.separation(SkyCoord(d["ra"] * u.deg, d["dec"] * u.deg)).deg < tol_deg:
            files.update({x["key"]: x for x in d["files"]})
    return list(files.values())


def _run_query(tap, query, ra, dec):
    """IRSA a veces devuelve 502 si el query síncrono tarda. Se intenta en orden:
    1) TAP asíncrono (sin límite de tiempo del proxy), 2) TAP síncrono, 3) SIA2."""
    import time

    for nombre, run in (("TAP async", tap.run_async), ("TAP sync", tap.search)):
        for intento in range(2):
            try:
                print(f"Consultando IRSA ({nombre})...")
                return run(query).to_table()
            except Exception as err:
                print(f"  falló: {type(err).__name__}: {str(err)[:120]}")
                time.sleep(5)

    for coleccion in ("spherex_qr2", "spherex_qr"):
        try:
            print(f"Consultando IRSA (SIA2, {coleccion})...")
            sia = pyvo.dal.SIA2Service("https://irsa.ipac.caltech.edu/SIA")
            t = sia.search(pos=(ra, dec, 0.001), collection=coleccion).to_table()
            if len(t):
                t = t["access_url", "t_min"]
                t.rename_columns(["access_url", "t_min"], ["uri", "t"])
                t.sort("t")
                return t
        except Exception as err:
            print(f"  falló: {type(err).__name__}: {str(err)[:120]}")
    raise RuntimeError("IRSA no respondió por ninguna vía; reintentar en unos minutos.")


def assign_epochs(exposures: list[dict]) -> None:
    """Agrega 'epoch' a cada exposición (la lista ya viene ordenada por tiempo)."""
    epoch = 0
    for i, e in enumerate(exposures):
        if i > 0 and e["t"] - exposures[i - 1]["t"] > EPOCH_GAP_DAYS:
            epoch += 1
        e["epoch"] = epoch


if __name__ == "__main__":
    if len(sys.argv) == 2:
        ra, dec = resolve(sys.argv[1])
    else:
        ra, dec = float(sys.argv[1]), float(sys.argv[2])
    print(f"RA={ra:.5f}  Dec={dec:.5f}")
    exps = find_exposures(ra, dec)
    print(f"{len(exps)} exposiciones")
    for ep in sorted({e["epoch"] for e in exps}):
        grupo = [e for e in exps if e["epoch"] == ep]
        fecha = Time(grupo[0]["t"], format="mjd").iso[:10]
        dets = sorted({e["detector"] for e in grupo})
        print(f"  época {ep}: {len(grupo)} exposiciones, desde {fecha}, detectores {dets}")
    if exps:
        print("Ejemplo de URL:", exps[0]["url"])
