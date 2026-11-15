"""Datos SINTÉTICOS con el formato del contrato, para trabajar mientras IRSA no responde.

Simula 3 épocas de un campo con estrellas fijas y una que se mueve ~10"/año
(como Barnard), con píxeles de 6.15", orientaciones distintas y ruido.

Uso:
    python make_fake.py            -> escribe FITS en data/sample/
En código:
    from make_fake import fake_exposures
    exposures = fake_exposures()   -> list[dict] igual que download.get_exposures()
"""
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

OUT = Path(__file__).resolve().parent.parent / "data" / "sample"
PIX = 6.15 / 3600          # grados por píxel
N = 31                     # tamaño del recorte en píxeles
RA0, DEC0 = 269.4463, 4.7677
PM = (-0.80, 10.36)        # movimiento propio en "/año (RA*cos(dec), Dec)
EPOCH_MJD = [60800.0, 60980.0, 61160.0]   # ~ may 2025, nov 2025, may 2026


def _wcs(angle_deg: float, dx: float, dy: float, ra0: float = RA0, dec0: float = DEC0) -> WCS:
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [ra0, dec0]
    w.wcs.crpix = [N / 2 + dx, N / 2 + dy]
    a = np.radians(angle_deg)
    w.wcs.cd = PIX * np.array([[-np.cos(a), np.sin(a)], [np.sin(a), np.cos(a)]])
    return w


def fake_exposures(ra0: float = RA0, dec0: float = DEC0, n_per_epoch: int = 12, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    # estrellas fijas: (dRA", dDec", flujo)
    fijas = [(-45, 30, 400), (50, -40, 250), (20, 60, 150), (-60, -55, 300)]
    yy, xx = np.mgrid[0:N, 0:N]
    out = []
    for ep, mjd in enumerate(EPOCH_MJD):
        years = (mjd - EPOCH_MJD[0]) / 365.25
        movil = (PM[0] * years, PM[1] * years, 800)
        for k in range(n_per_epoch):
            det = k % 6 + 1
            w = _wcs(rng.uniform(0, 360), *rng.uniform(-0.5, 0.5, 2), ra0, dec0)
            img = rng.normal(0, 3.0, (N, N))
            for dra, ddec, flux in fijas + [movil]:
                ra = ra0 + dra / 3600 / np.cos(np.radians(dec0))
                dec = dec0 + ddec / 3600
                x, y = w.all_world2pix(ra, dec, 0)
                img += flux * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 0.8**2))
            flags = np.zeros((N, N), dtype="uint32")
            malos = rng.integers(0, N, (3, 2))          # unos píxeles malos
            flags[malos[:, 0], malos[:, 1]] = 1
            img[malos[:, 0], malos[:, 1]] += 5000
            out.append({
                "t": mjd + k * 0.1,
                "detector": det,
                "epoch": ep,
                "image": img.astype("float32"),
                "variance": np.full((N, N), 9.0, dtype="float32"),
                "flags": flags,
                "wcs": w,
                "path": str(OUT / f"fake_ep{ep}_d{det}_{k:02d}.fits"),
            })
    return out


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    exps = fake_exposures()
    for e in exps:
        hdr = e["wcs"].to_header()
        hdr["MJD-OBS"] = e["t"]
        hdr["DETECTOR"] = e["detector"]
        fits.HDUList([
            fits.PrimaryHDU(),
            fits.ImageHDU(e["image"], header=hdr, name="IMAGE"),
            fits.ImageHDU(e["flags"], header=hdr, name="FLAGS"),
            fits.ImageHDU(e["variance"], header=hdr, name="VARIANCE"),
        ]).writeto(e["path"], overwrite=True)
    print(f"{len(exps)} FITS sintéticos en {OUT}")
