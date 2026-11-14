"""Interruptor de la fuente de datos. El resto del código solo usa este archivo.

    SPHEREX_SOURCE=fake   -> datos sintéticos de make_fake.py   (por defecto)
    SPHEREX_SOURCE=irsa   -> datos reales: SIMBAD (nombres) + IRSA (imágenes)

En PowerShell:   $env:SPHEREX_SOURCE = "irsa"
En bash:         export SPHEREX_SOURCE=irsa
"""
import os

SOURCE = os.environ.get("SPHEREX_SOURCE", "fake").lower()


def resolve_name(name: str) -> tuple[float, float]:
    """Nombre -> (ra, dec). Los 5 objetos de prueba funcionan sin internet."""
    from query import resolve
    return resolve(name)


def get_exposures(ra: float, dec: float, fov: float = 0.05) -> list[dict]:
    """(ra, dec, fov) -> lista de exposiciones en el formato del contrato."""
    if SOURCE == "irsa":
        from download import get_exposures as real
        return real(ra, dec, fov)
    from make_fake import fake_exposures
    return fake_exposures(ra, dec)
