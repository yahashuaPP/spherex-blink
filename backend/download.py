"""Y2 + Y3 — Bajar recortes con cola de prioridad + caché, y leerlos al formato del contrato.

Uso rápido:
    python download.py "Barnard's Star"
    python download.py 269.446 4.768
"""
import asyncio
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

import httpx
from astropy.io import fits
from astropy.wcs import WCS

from query import find_exposures, resolve

CACHE = Path(__file__).resolve().parent.parent / "cache"
MAX_PARALLEL = 8      # descargas simultáneas (no subir mucho: IRSA puede bloquear)
MAX_PER_EPOCH = 30    # tope de exposiciones por época


def prioritize(exposures: list[dict], max_per_epoch: int = MAX_PER_EPOCH) -> list[tuple]:
    """Devuelve [(prioridad, i, exposición)] intercalando épocas y detectores.

    Dentro de cada época se rota entre detectores (1 de cada uno, luego otra vuelta).
    La prioridad es el puesto dentro de su época: así primero baja 1 exposición
    de cada época, luego la 2.ª de cada una, etc. El video existe desde el inicio
    y va mejorando.
    """
    by_epoch = defaultdict(lambda: defaultdict(list))
    for e in exposures:
        by_epoch[e["epoch"]][e["detector"]].append(e)

    items = []
    for epoch, by_det in by_epoch.items():
        rank = 0
        colas = [list(v) for _, v in sorted(by_det.items())]
        while any(colas) and rank < max_per_epoch:
            for cola in colas:
                if cola and rank < max_per_epoch:
                    items.append((rank, epoch, cola.pop(0)))
                    rank += 1
    items.sort(key=lambda x: (x[0], x[1]))
    return [(rank, i, e) for i, (rank, _, e) in enumerate(items)]


def cache_path(e: dict, ra: float, dec: float, fov: float) -> Path:
    carpeta = CACHE / f"{ra:.4f}_{dec:.4f}_{fov}"
    carpeta.mkdir(parents=True, exist_ok=True)
    nombre = hashlib.md5(e["url"].encode()).hexdigest()[:16]
    return carpeta / f"ep{e['epoch']}_d{e['detector']}_{nombre}.fits"


async def download_all(exposures, ra, dec, fov, on_done=None) -> list[dict]:
    """Baja todo respetando la prioridad. Devuelve las exposiciones con 'path'.

    on_done(exposición) se llama cada vez que termina una: sirve para que el
    API vaya avisando al frontend (carga progresiva).
    """
    queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
    for item in prioritize(exposures):
        queue.put_nowait(item)

    listas = []

    async def worker(client: httpx.AsyncClient):
        while True:
            try:
                _, _, e = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            path = cache_path(e, ra, dec, fov)
            if not path.exists():
                try:
                    r = await client.get(e["url"])
                    r.raise_for_status()
                    tmp = path.with_suffix(".part")
                    tmp.write_bytes(r.content)
                    tmp.rename(path)
                except Exception as err:  # una que falle no tumba el resto
                    print(f"  falló {e['url'][-60:]}: {err}")
                    continue
            e["path"] = str(path)
            listas.append(e)
            if on_done:
                on_done(e)

    async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
        await asyncio.gather(*(worker(client) for _ in range(MAX_PARALLEL)))
    return sorted(listas, key=lambda e: e["t"])


def load_exposure(e: dict) -> dict:
    """FITS -> dict del CONTRATO con Ezequiel."""
    with fits.open(e["path"]) as hdul:
        return {
            "t": e["t"],
            "detector": e["detector"],
            "epoch": e["epoch"],
            "image": hdul["IMAGE"].data.astype("float32"),
            "variance": hdul["VARIANCE"].data.astype("float32"),
            "flags": hdul["FLAGS"].data.astype("uint32"),  # science.py usa máscaras de 32 bits sin signo
            "wcs": WCS(hdul["IMAGE"].header),
            "path": e["path"],
        }


def get_exposures(ra: float, dec: float, fov: float = 0.05) -> list[dict]:
    """Todo junto: buscar -> bajar -> leer. Esto es lo que usa science.py."""
    exps = find_exposures(ra, dec, fov)
    bajadas = asyncio.run(download_all(exps, ra, dec, fov))
    return [load_exposure(e) for e in bajadas]


if __name__ == "__main__":
    if len(sys.argv) == 2:
        ra, dec = resolve(sys.argv[1])
    else:
        ra, dec = float(sys.argv[1]), float(sys.argv[2])
    data = get_exposures(ra, dec)
    print(f"{len(data)} exposiciones listas")
    if data:
        d = data[0]
        print("image:", d["image"].shape, "| detector:", d["detector"], "| época:", d["epoch"])
        with fits.open(d["path"]) as h:
            h.info()
