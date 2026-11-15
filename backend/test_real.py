"""Prueba con UN recorte REAL de SPHEREx (no necesita el buscador de IRSA).

El servicio de recortes de IRSA sí responde aunque el buscador (TAP/SIA) esté caído.
Este script baja un recorte de un archivo conocido (el del tutorial oficial) y revisa
que nuestro código lo sepa leer.

Uso (parado en backend/):
    python test_real.py
"""
import sys
from pathlib import Path

import httpx
import numpy as np
from astropy.io import fits

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "science"))
import science as sc  # noqa: E402

RA, DEC, SIZE = 305.59875, 41.14889, 0.1
URL = ("https://irsa.ipac.caltech.edu/ibe/data/spherex/qr2/level2/2025W18_1B/"
       "l2b-v20-2025-240/4/level2_2025W18_1B_0023_2D4_spx_l2b-v20-2025-240.fits"
       f"?center={RA},{DEC}d&size={SIZE}")
OUT = ROOT / "data" / "real_test.fits"

OUT.parent.mkdir(parents=True, exist_ok=True)
print("Bajando recorte real...")
r = httpx.get(URL, timeout=120, follow_redirects=True)
r.raise_for_status()
OUT.write_bytes(r.content)
print(f"OK: {len(r.content) / 1e6:.2f} MB -> {OUT}\n")

with fits.open(OUT) as h:
    h.info()
    hdr = h["IMAGE"].header
    print("\nBUNIT:", hdr.get("BUNIT"), "| DETECTOR:", hdr.get("DETECTOR"),
          "| MJD-AVG:", hdr.get("MJD-AVG"), "| MJD-OBS:", hdr.get("MJD-OBS"))
    print("Claves de tiempo/WCS-W:", [k for k in hdr if "MJD" in k or "DATE" in k or k.endswith("W")][:20])
    img, flags = h["IMAGE"].data, h["FLAGS"].data.astype("uint32")
    print("\nimage:", img.shape, img.dtype, "| flags:", flags.dtype)

    print("\nBits de FLAGS (porcentaje de píxeles con cada bit):")
    for bit in range(32):
        frac = np.mean((flags >> bit) & 1)
        if frac > 0:
            print(f"  bit {bit:2d}: {100 * frac:6.2f} %")
    iy, ix = np.unravel_index(np.nanargmax(img), img.shape)
    f = int(flags[iy, ix])
    print(f"\nPíxel más brillante ({ix},{iy}) = {img[iy, ix]:.1f}; "
          f"bits encendidos ahí: {[b for b in range(32) if (f >> b) & 1]}")
    print(f"Píxeles que science.py descartaría con la máscara actual: "
          f"{100 * np.mean((flags & sc.DEFAULT_BAD_FLAG_MASK) != 0):.1f} %")

print("\nLeyendo con science.load_exposure_from_fits ...")
e = sc.load_exposure_from_fits(OUT)
print("t =", e["t"], "| detector =", e["detector"], "| image", e["image"].shape)
ny, nx = e["image"].shape
print("longitud de onda en el centro (µm, ancho):", sc.extract_wavelength_um(e, nx / 2, ny / 2))
res = sc.build_frames([e], RA, DEC, SIZE)
print("build_frames: épocas =", len(res["epochs"]),
      "| píxeles válidos en el co-add:",
      int(np.isfinite(res["epochs"][0]["image"]).sum()) if res["epochs"] else 0)
print("\nListo. Pega toda esta salida en el chat.")
