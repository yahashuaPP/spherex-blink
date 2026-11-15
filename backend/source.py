"""Interruptor de la fuente de datos. El resto del código solo usa este archivo.

    SPHEREX_SOURCE=fake   -> datos SINTÉTICOS (por defecto)
    SPHEREX_SOURCE=irsa   -> datos reales: índices propios + recortes desde Amazon S3
    SPHEREX_SOURCE=demo   -> demo pública: solo sirve lo ya calculado en precomputed/

En PowerShell:   $env:SPHEREX_SOURCE = "irsa"
En bash:         export SPHEREX_SOURCE=irsa
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "science") not in sys.path:
    sys.path.insert(0, str(ROOT / "science"))

SOURCE = os.environ.get("SPHEREX_SOURCE", "fake").lower()


def resolve_name(name: str) -> tuple[float, float]:
    """Nombre -> (ra, dec). Los 5 objetos de prueba funcionan sin internet."""
    from query import resolve
    return resolve(name)


def get_exposures(ra: float, dec: float, fov: float = 0.1) -> list[dict]:
    """(ra, dec, fov) -> lista de exposiciones en el formato del contrato."""
    if SOURCE == "irsa":
        from download import get_exposures as real
        return real(ra, dec, fov)

    # --- Modo sintético ---
    import science as sc

    known = sc.match_known_target(ra, dec, tol_deg=0.05)
    if known is not None:
        # Simulador de Ezequiel: 6 detectores, mapa de longitud de onda y espectro modelo.
        from validate_and_test import create_sample_spherex_fits

        paths = create_sample_spherex_fits(known["key"], ROOT / "data" / "sample")
        return [sc.load_exposure_from_fits(p) for p in paths]

    from make_fake import fake_exposures  # cualquier otra coordenada: campo genérico
    return fake_exposures(ra, dec)
