"""Índice espacial de TODO el cielo de SPHEREx.

Construcción (una sola vez; se puede cortar y retomar):
    python sky_index.py
Guarda en backend/index/sky/ un archivo por media semana con el apuntado de cada
exposición. Va saltando entre semanas (no en orden de fecha) para que, aunque se
corte a medias, lo indexado ya cubra toda la misión con varias épocas.

Consulta de prueba:
    python sky_index.py 269.4463 4.7677

Estructura: una tabla plana (arreglos de numpy) con (ra, dec) de cada exposición.
Buscar una coordenada es una resta vectorizada sobre ~280 000 filas: milisegundos.
Después solo se leen las cabeceras de las pocas exposiciones candidatas.
"""
import asyncio
import json
import sys
import time

import httpx
import numpy as np
from astropy.wcs import WCS

import s3_index as s3

SKY = s3.INDEX / "sky"
PARALLEL = 96
_table = {"n": -1}


# ------------------------------------------------------------ construcción
async def list_weeks(client):
    chosen = {}
    for root in s3.ROOTS:
        prefixes, _ = await s3.s3_list(client, f"{root}/level2/")
        for p in prefixes:
            chosen.setdefault(p.split("/")[-2][:9], p)
    return sorted(chosen.values(), key=lambda p: p.split("/")[-2])


def spread(n):
    """Orden 0, n/2, n/4, 3n/4...: lo indexado primero queda repartido en el tiempo."""
    bits = max(n - 1, 1).bit_length()
    return sorted(range(n), key=lambda i: int(format(i, f"0{bits}b")[::-1], 2))


async def build():
    SKY.mkdir(parents=True, exist_ok=True)
    scout = s3.load_json(s3.SCOUT_FILE, {})
    sem = asyncio.Semaphore(PARALLEL)
    async with httpx.AsyncClient(timeout=60, limits=httpx.Limits(max_connections=PARALLEL + 10)) as client:
        weeks = await list_weeks(client)
        todo = [weeks[i] for i in spread(len(weeks)) if not (SKY / f"{weeks[i].split('/')[-2]}.json").exists()]
        print(f"{len(weeks)} medias semanas en S3; faltan {len(todo)} por indexar.")
        t0 = time.time()
        for n, w in enumerate(todo, 1):
            files = await s3.week_files(client, w)

            async def one(obsid, dets):
                if obsid not in scout:
                    d = next(x for x in (2, 5, 1, 3, 4, 6) if x in dets)
                    try:
                        scout[obsid] = [*(await s3.read_pointing(client, sem, dets[d])), d]
                    except Exception:
                        pass

            await asyncio.gather(*(one(o, d) for o, d in files.items()))

            versions, obs, odd = [], [], {}
            for obsid, dets in files.items():
                if obsid not in scout:
                    continue
                vidx = [-1] * 6
                for d, key in dets.items():
                    v = key.split("/")[3]
                    if key != f"{w}{v}/{d}/level2_{obsid}D{d}_spx_{v}.fits":
                        odd[f"{obsid}:{d}"] = key          # nombre fuera del patrón: se guarda entero
                        continue
                    if v not in versions:
                        versions.append(v)
                    vidx[d - 1] = versions.index(v)
                ra, dec, sd = scout[obsid]
                obs.append([obsid, round(ra, 4), round(dec, 4), sd, vidx])
            (SKY / f"{w.split('/')[-2]}.json").write_text(
                json.dumps({"prefix": w, "versions": versions, "obs": obs, "odd": odd}))
            falta = (time.time() - t0) / n * (len(todo) - n) / 60
            print(f"  [{n}/{len(todo)}] {w.split('/')[-2]}: {len(obs)} exposiciones. "
                  f"Faltan ~{falta:.0f} min.", flush=True)
    print("Índice completo.")


# ------------------------------------------------------------ consulta
def n_weeks() -> int:
    return len(list(SKY.glob("*.json"))) if SKY.exists() else 0


def load_table():
    """Tabla plana en memoria; se recarga sola si el índice creció."""
    n = n_weeks()
    if _table["n"] != n:
        weeks, ra, dec, sd, wi, oi = [], [], [], [], [], []
        for i, f in enumerate(sorted(SKY.glob("*.json"))):
            w = json.loads(f.read_text())
            weeks.append(w)
            for j, o in enumerate(w["obs"]):
                ra.append(o[1]); dec.append(o[2]); sd.append(o[3]); wi.append(i); oi.append(j)
        _table.update(n=n, weeks=weeks, ra=np.radians(ra), dec=np.radians(dec),
                      sd=np.array(sd), wi=np.array(wi), oi=np.array(oi))
    return _table


def candidates(ra: float, dec: float) -> list[str]:
    """Archivos (de los 6 detectores) de las exposiciones que apuntaban cerca. Sin red."""
    t = load_table()
    if not t["n"]:
        return []
    a, d = np.radians(ra), np.radians(dec)
    c = np.sin(t["dec"]) * np.sin(d) + np.cos(t["dec"]) * np.cos(d) * np.cos(t["ra"] - a)
    sep = np.degrees(np.arccos(np.clip(c, -1, 1)))
    near = np.where(sep < np.where(np.isin(t["sd"], (2, 5)), 6.5, 10.0))[0]
    keys = []
    for k in near:
        w = t["weeks"][t["wi"][k]]
        obsid, _, _, _, vidx = w["obs"][t["oi"][k]]
        for det in range(1, 7):
            if vidx[det - 1] >= 0:
                v = w["versions"][vidx[det - 1]]
                keys.append(f"{w['prefix']}{v}/{det}/level2_{obsid}D{det}_spx_{v}.fits")
            elif f"{obsid}:{det}" in w.get("odd", {}):
                keys.append(w["odd"][f"{obsid}:{det}"])
    return keys


async def _covers(client, sem, key, ra, dec):
    """¿Este archivo contiene la posición? Primero una lectura chica, luego la cabecera completa."""
    try:
        cra, cdec = await s3.read_pointing(client, sem, key)
        if s3.sep_deg(cra, cdec, ra, dec) > 2.7:
            return None
        h = await s3.read_header(client, sem, key)
        x, y = WCS(h).all_world2pix(ra, dec, 0, quiet=True)
        if 5 <= x <= s3.NPIX - 5 and 5 <= y <= s3.NPIX - 5:
            return {"key": key, "t": float(h["MJD-AVG"]), "detector": int(h["DETECTOR"])}
    except Exception:
        return None


async def _find(ra, dec):
    keys = candidates(ra, dec)
    sem = asyncio.Semaphore(PARALLEL)
    async with httpx.AsyncClient(timeout=60, limits=httpx.Limits(max_connections=PARALLEL + 10)) as client:
        hits = await asyncio.gather(*(_covers(client, sem, k, ra, dec) for k in keys))
    return sorted((h for h in hits if h), key=lambda h: h["t"])


def find(ra: float, dec: float) -> list[dict]:
    """Archivos que cubren (ra, dec). Guarda el resultado para no repetir la consulta."""
    n = n_weeks()
    if not n:
        return []
    cache = s3.INDEX / f"pos_{ra:.4f}_{dec:.4f}.json"
    d = s3.load_json(cache, {})
    if d.get("sky_weeks") != n:
        d = {"ra": ra, "dec": dec, "sky_weeks": n, "files": asyncio.run(_find(ra, dec))}
        cache.write_text(json.dumps(d, indent=1))
    return d["files"]


if __name__ == "__main__":
    if len(sys.argv) == 3:
        ra, dec = float(sys.argv[1]), float(sys.argv[2])
        t0 = time.time()
        print(f"{n_weeks()} medias semanas indexadas; {len(candidates(ra, dec))} archivos candidatos "
              f"({1000 * (time.time() - t0):.0f} ms)")
        files = find(ra, dec)
        t = np.array([f["t"] for f in files])
        print(f"{len(files)} archivos cubren la posición, en "
              f"{1 + int(np.sum(np.diff(t) > 30)) if len(t) else 0} época(s). {time.time() - t0:.1f} s en total.")
    else:
        try:
            asyncio.run(build())
        except KeyboardInterrupt:
            print("\nCortado. Lo indexado quedó guardado; vuelve a correrlo para seguir.")
