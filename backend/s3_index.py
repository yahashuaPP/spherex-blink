"""Buscador propio para cuando el buscador de IRSA (TAP/SIA) está caído.

Los archivos de SPHEREx también están en un bucket público de Amazon S3. Este script
lee solo la cabecera (28 KB) de cada exposición para saber a qué parte del cielo apunta,
y guarda en backend/index/<objeto>.json la lista de archivos que cubren cada objeto.
Con ese índice, query.py y download.py ya funcionan sin el buscador de IRSA.

Uso (parado en backend/):
    python s3_index.py                      # los 5 objetos de prueba, Barnard primero
    python s3_index.py "Barnard's Star"     # uno solo
    python s3_index.py 269.4463 4.7677      # una coordenada cualquiera

Se puede cortar con Ctrl+C y volver a correr: recuerda lo ya revisado.
"""
import asyncio
import datetime as dt
import json
import re
import sys
import warnings
from pathlib import Path

import httpx
import numpy as np
from astropy import log
from astropy.io import fits
from astropy.wcs import WCS

from query import KNOWN, resolve

warnings.filterwarnings("ignore")
log.setLevel("ERROR")

BUCKET = "https://nasa-irsa-spherex.s3.amazonaws.com"
ROOTS = ["qr3", "qr2"]            # si una semana está en las dos entregas, gana la primera
INDEX = Path(__file__).resolve().parent / "index"
SCOUT_FILE = INDEX / "_scout.json"
HEADER_BYTES = 28800              # cabecera primaria + cabecera de IMAGE (comprobado: 22 KB)
PARALLEL = 48
NPIX = 2040
MAX_WEEKS = 30                    # tope de medias semanas a revisar por objeto
ELONG_TOL = 14.0                  # grados de margen alrededor de 90° del Sol
NAMES = ["Barnard's Star", "RW Aurigae", "Epsilon Indi B", "WISE 0855", "R Draconis"]


# ---------------------------------------------------------------- geometría
def ecliptic(ra, dec):
    """(ra, dec) -> (longitud, latitud) eclípticas en grados."""
    e = np.radians(23.4393)
    a, d = np.radians(ra), np.radians(dec)
    lat = np.arcsin(np.sin(d) * np.cos(e) - np.cos(d) * np.sin(e) * np.sin(a))
    lon = np.arctan2(np.sin(a) * np.cos(e) + np.tan(d) * np.sin(e), np.cos(a))
    return float(np.degrees(lon) % 360), float(np.degrees(lat))


def sun_lon(day: dt.date) -> float:
    n = (day - dt.date(2000, 1, 1)).days - 0.5
    g = np.radians((357.528 + 0.9856003 * n) % 360)
    return float((280.460 + 0.9856474 * n + 1.915 * np.sin(g) + 0.020 * np.sin(2 * g)) % 360)


def visible_days(ra, dec, start=dt.date(2025, 4, 20), end=None) -> set:
    """Días en que SPHEREx puede ver el objeto: apunta a ~90° del Sol.

    Margen de 14°: una exposición real conocida (2025-04-28) estaba a 80.7° del Sol.
    """
    end = end or dt.date.today()
    lon, lat = ecliptic(ra, dec)
    days = set()
    d = start
    while d <= end:
        c = np.cos(np.radians(lat)) * np.cos(np.radians(sun_lon(d) - lon))
        if abs(np.degrees(np.arccos(c)) - 90.0) < ELONG_TOL:
            days.add(d)
        d += dt.timedelta(days=1)
    return days


def sep_deg(ra1, dec1, ra2, dec2) -> float:
    a1, d1, a2, d2 = map(np.radians, (ra1, dec1, ra2, dec2))
    c = np.sin(d1) * np.sin(d2) + np.cos(d1) * np.cos(d2) * np.cos(a1 - a2)
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


# ---------------------------------------------------------------- S3
async def s3_list(client, prefix, delimiter="/"):
    """Devuelve (subcarpetas, archivos) de un prefijo, siguiendo la paginación."""
    prefixes, keys, token = [], [], None
    while True:
        params = {"list-type": "2", "prefix": prefix}
        if delimiter:
            params["delimiter"] = delimiter
        if token:
            params["continuation-token"] = token
        r = await get(client, BUCKET + "/", params=params)
        prefixes += re.findall(r"<CommonPrefixes><Prefix>([^<]*)</Prefix>", r.text)
        keys += re.findall(r"<Key>([^<]*)</Key>", r.text)
        m = re.search(r"<NextContinuationToken>([^<]*)</NextContinuationToken>", r.text)
        if not m:
            return prefixes, keys
        token = m.group(1)


async def get(client, url, tries=4, **kw):
    for i in range(tries):
        try:
            r = await client.get(url, **kw)
            if r.status_code in (200, 206):
                return r
        except httpx.HTTPError:
            pass
        await asyncio.sleep(1.5 * (i + 1))
    raise RuntimeError(f"No pude leer {url}")


async def read_header(client, sem, key):
    """Cabecera de la extensión IMAGE de un archivo, leyendo solo el inicio."""
    async with sem:
        r = await get(client, f"{BUCKET}/{key}", headers={"Range": f"bytes=0-{HEADER_BYTES - 1}"})
    return fits.Header.fromstring(r.content[2880:].decode("ascii", "replace"))


async def read_pointing(client, sem, key):
    """(ra, dec) del centro de la imagen. Lee solo 8 KB: CRVAL1/2 están entre los bytes ~10300 y ~11400."""
    async with sem:
        r = await get(client, f"{BUCKET}/{key}", headers={"Range": "bytes=8000-15999"})
    m1 = re.search(rb"CRVAL1  =\s*([-+0-9.eE]+)", r.content)
    m2 = re.search(rb"CRVAL2  =\s*([-+0-9.eE]+)", r.content)
    if m1 and m2:
        return float(m1.group(1)), float(m2.group(1))
    h = await read_header(client, sem, key)          # por si la cabecera cambia de tamaño
    return float(h["CRVAL1"]), float(h["CRVAL2"])


def week_monday(name: str):
    m = re.match(r"(\d{4})W(\d{2})_", name)
    return dt.date.fromisocalendar(int(m.group(1)), int(m.group(2)), 1) if m else None


async def week_files(client, week_prefix):
    """{obsid: {detector: key}} de una media semana (se queda con la versión más nueva)."""
    out = {}
    versions, _ = await s3_list(client, week_prefix)
    for v in sorted(versions):
        _, keys = await s3_list(client, v, delimiter=None)
        for k in keys:
            m = re.search(r"level2_(.+)D(\d)_spx_.*\.fits$", k)
            if m:
                out.setdefault(m.group(1), {})[int(m.group(2))] = k
    return out


# ---------------------------------------------------------------- índice
def load_json(path, default):
    return json.loads(path.read_text()) if path.exists() else default


def slug(name):
    return re.sub(r"[^a-z0-9]", "", name.lower()) or "objeto"


async def index_target(client, sem, name, ra, dec, weeks, scout):
    path = INDEX / f"{slug(name)}.json"
    idx = load_json(path, {"name": name, "ra": ra, "dec": dec, "weeks_done": [], "files": []})
    days = visible_days(ra, dec)
    wanted = [w for w in weeks
              if any(week_monday(w.split("/")[-2]) + dt.timedelta(days=i) in days for i in range(7))]
    if len(wanted) > MAX_WEEKS:      # objetos cerca del polo eclíptico se ven casi siempre: muestreo
        paso = len(wanted) / MAX_WEEKS
        wanted = [wanted[int(i * paso)] for i in range(MAX_WEEKS)]
    wanted = [w for w in wanted if w not in idx["weeks_done"]]
    print(f"\n=== {name}  RA={ra:.4f} Dec={dec:.4f} | {len(wanted)} medias semanas por revisar, "
          f"{len(idx['files'])} archivos ya encontrados")

    for w in wanted:
        files = await week_files(client, w)

        # 1) Un archivo "explorador" por exposición: ¿hacia dónde apuntaba?
        async def scout_one(obsid, dets):
            if obsid in scout:
                return
            d = next(x for x in (2, 5, 1, 3, 4, 6) if x in dets)
            try:
                scout[obsid] = [*(await read_pointing(client, sem, dets[d])), d]
            except Exception:
                pass

        await asyncio.gather(*(scout_one(o, d) for o, d in files.items()))

        # 2) Las exposiciones cercanas se revisan detector por detector, con el WCS completo.
        cerca = [o for o in files if o in scout
                 and sep_deg(scout[o][0], scout[o][1], ra, dec) < (6.5 if scout[o][2] in (2, 5) else 10.0)]

        async def check(key):
            try:
                h = await read_header(client, sem, key)
                if sep_deg(float(h["CRVAL1"]), float(h["CRVAL2"]), ra, dec) > 2.7:
                    return None
                x, y = WCS(h).all_world2pix(ra, dec, 0, quiet=True)
                if 5 <= x <= NPIX - 5 and 5 <= y <= NPIX - 5:
                    return {"key": key, "t": float(h["MJD-AVG"]), "detector": int(h["DETECTOR"])}
            except Exception:
                return None

        hits = [h for h in await asyncio.gather(*(check(k) for o in cerca for k in files[o].values())) if h]
        idx["files"] += hits
        idx["weeks_done"].append(w)
        idx["files"].sort(key=lambda f: f["t"])
        path.write_text(json.dumps(idx, indent=1))
        SCOUT_FILE.write_text(json.dumps(scout))
        print(f"  {w.split('/')[0]} {w.split('/')[-2]}: {len(files)} exposiciones, "
              f"{len(cerca)} cercanas, {len(hits)} archivos cubren el objeto (total {len(idx['files'])})")
    return idx


async def main(targets):
    INDEX.mkdir(exist_ok=True)
    scout = load_json(SCOUT_FILE, {})
    sem = asyncio.Semaphore(PARALLEL)
    limits = httpx.Limits(max_connections=PARALLEL + 10)
    async with httpx.AsyncClient(timeout=60, limits=limits) as client:
        chosen = {}
        for root in ROOTS:
            prefixes, _ = await s3_list(client, f"{root}/level2/")
            for p in prefixes:
                chosen.setdefault(p.split("/")[-2][:9], p)   # 2025W18_1 -> gana la primera entrega
        weeks = list(chosen.values())
        weeks.sort(key=lambda p: p.split("/")[-2])
        print(f"{len(weeks)} medias semanas de datos en S3 "
              f"({weeks[0].split('/')[-2]} a {weeks[-1].split('/')[-2]})")
        for name, ra, dec in targets:
            idx = await index_target(client, sem, name, ra, dec, weeks, scout)
            t = np.array([f["t"] for f in idx["files"]])
            if len(t):
                epocas = 1 + int(np.sum(np.diff(t) > 30))
                print(f"  -> {name}: {len(t)} archivos en {epocas} época(s). Guardado en index/{slug(name)}.json")
            else:
                print(f"  -> {name}: no encontré archivos. Avísale a Claude.")


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) == 2:
        tg = [(f"{float(a[0]):.4f}_{float(a[1]):.4f}", float(a[0]), float(a[1]))]
    else:
        tg = [(n, *resolve(n)) for n in ([a[0]] if a else NAMES)]
    try:
        asyncio.run(main(tg))
    except KeyboardInterrupt:
        print("\nCortado. Lo ya revisado quedó guardado; vuelve a correrlo para seguir.")
