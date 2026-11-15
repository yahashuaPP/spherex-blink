"""Recortes de SPHEREx hechos por nosotros, leyendo solo los bytes necesarios desde Amazon S3.

No depende de IRSA. Cada archivo original pesa 71 MB; un recorte de 6' baja ~2 MB
(las filas que tocan el recorte de IMAGE, FLAGS, VARIANCE y ZODI, más la tabla de
longitudes de onda). El cubo de PSF (5 MB) se omite porque no lo usamos.

El resultado es un FITS con la misma estructura que los recortes de IRSA:
PRIMARY, IMAGE, FLAGS, VARIANCE, ZODI, WCS-WAVE.
"""
import warnings

import numpy as np
from astropy import log
from astropy.io import fits
from astropy.wcs import WCS

warnings.filterwarnings("ignore")
log.setLevel("ERROR")

BUCKET = "https://nasa-irsa-spherex.s3.amazonaws.com"
BLOCK = 2880
PIX_DEG = 6.15 / 3600
CUT_HDUS = ("IMAGE", "FLAGS", "VARIANCE", "ZODI")


def _pad(n: int) -> int:
    return (n + BLOCK - 1) // BLOCK * BLOCK


async def _read_header(read, offset: int):
    """Lee la cabecera que empieza en `offset`. Devuelve (Header, bytes que ocupa)."""
    raw, size = b"", 28800
    while True:
        raw = await read(offset, size)
        for i in range(0, len(raw) - 79, 80):
            if raw[i:i + 8] == b"END     ":
                hlen = _pad(i + 80)
                return fits.Header.fromstring(raw[:hlen].decode("ascii", "replace")), hlen
        if len(raw) < size or size > 300000:
            raise RuntimeError("No encontré el final de la cabecera")
        size *= 2


def _data_bytes(h) -> int:
    n = 1
    for k in range(1, h.get("NAXIS", 0) + 1):
        n *= h[f"NAXIS{k}"]
    if h.get("NAXIS", 0) == 0:
        n = 0
    return _pad(abs(h["BITPIX"]) // 8 * h.get("GCOUNT", 1) * (h.get("PCOUNT", 0) + n))


async def make_cutout(read, ra: float, dec: float, fov_deg: float) -> bytes:
    """`read(offset, nbytes)` es una función async que devuelve bytes del archivo original."""
    npix = max(int(round(fov_deg / PIX_DEG)), 8)
    out = [await read(0, BLOCK)]                       # cabecera primaria tal cual
    offset, window = BLOCK, None

    while True:
        try:
            h, hlen = await _read_header(read, offset)
        except Exception:
            break                                       # fin del archivo
        name = str(h.get("EXTNAME", "")).strip().upper()
        dlen = _data_bytes(h)
        dstart = offset + hlen

        if name in CUT_HDUS:
            nx, ny = h["NAXIS1"], h["NAXIS2"]
            if window is None:                          # se decide una vez, con el WCS de IMAGE
                x, y = WCS(h).all_world2pix(ra, dec, 0, quiet=True)
                if not (0 <= x < nx and 0 <= y < ny):
                    raise ValueError("La posición cae fuera de esta imagen")
                x0 = int(np.clip(round(float(x)) - npix // 2, 0, max(nx - npix, 0)))
                y0 = int(np.clip(round(float(y)) - npix // 2, 0, max(ny - npix, 0)))
                window = (x0, y0, min(x0 + npix, nx), min(y0 + npix, ny))
            x0, y0, x1, y1 = window
            bpp = abs(h["BITPIX"]) // 8
            dtype = {-32: ">f4", -64: ">f8", 32: ">i4", 16: ">i2", 8: "u1", 64: ">i8"}[h["BITPIX"]]
            rows = await read(dstart + y0 * nx * bpp, (y1 - y0) * nx * bpp)
            cut = np.frombuffer(rows, dtype=dtype).reshape(y1 - y0, nx)[:, x0:x1]

            h["NAXIS1"], h["NAXIS2"] = x1 - x0, y1 - y0
            for alt in [""] + [chr(c) for c in range(65, 91)]:      # WCS principal y alternos (A, W...)
                for ax, shift in (("1", x0), ("2", y0)):
                    k = f"CRPIX{ax}{alt}"
                    if k in h:
                        h[k] = h[k] - shift
            body = cut.tobytes()
            out += [h.tostring(sep="", endcard=True, padding=True).encode("ascii"),
                    body + b"\0" * (_pad(len(body)) - len(body))]
        elif name == "WCS-WAVE":                        # tabla chica: se copia entera
            out.append(await read(offset, hlen + dlen))
        offset = dstart + dlen                          # PSF y demás: se saltan

    if window is None:
        raise RuntimeError("El archivo no tiene extensión IMAGE")
    return b"".join(out)


def s3_reader(client, key: str, sem=None, tries: int = 4):
    """Lector por rangos de bytes sobre un archivo del bucket público."""
    import asyncio

    async def read(offset: int, n: int) -> bytes:
        for i in range(tries):
            try:
                if sem:
                    async with sem:
                        r = await client.get(f"{BUCKET}/{key}", headers={"Range": f"bytes={offset}-{offset + n - 1}"})
                else:
                    r = await client.get(f"{BUCKET}/{key}", headers={"Range": f"bytes={offset}-{offset + n - 1}"})
                if r.status_code == 206:
                    return r.content
                if r.status_code == 416:                # pedimos más allá del final del archivo
                    return b""
            except Exception:
                pass
            await asyncio.sleep(1.0 * (i + 1))
        raise RuntimeError(f"No pude leer {key}")

    return read
