"""Orquestador: una corrida completa por objeto, de principio a fin.

    python pipeline.py                      # los 5 objetos de prueba
    python pipeline.py "Barnard's Star"     # uno solo
    python pipeline.py 150.1 2.2            # una coordenada
    python pipeline.py --status             # solo muestra en qué está cada cosa
    python pipeline.py --force ...          # recalcula aunque ya exista el resultado

Pasos por objeto (cada uno se salta si ya está hecho):
    1. nombre -> coordenadas
    2. buscar qué exposiciones cubren el objeto (índices propios)
    3. bajar los recortes desde Amazon S3 (con caché)
    4. ciencia: épocas, resta, suma, bandas, posición por exposición
    5. guardar el resultado en precomputed/ -> la web lo sirve al instante, sin red

Si un objeto falla, sigue con el siguiente y al final dice cuáles fallaron y en qué paso.
Siempre usa datos REALES; el modo sintético es solo para la web sin preparar.
"""
import json
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import source  # noqa: E402

FOV = 0.1
NAMES = ["Barnard's Star", "Epsilon Indi B", "WISE 0855", "RW Aurigae", "R Draconis"]


def targets_from_args(args):
    if len(args) == 2:
        try:
            ra, dec = float(args[0]), float(args[1])
            return [(f"{ra:.4f} {dec:+.4f}", ra, dec)]
        except ValueError:
            pass
    return [(n, None, None) for n in (args or NAMES)]


def status():
    import query
    import sky_index
    from api import pre_path

    print(f"Índice de todo el cielo: {sky_index.n_weeks()} medias semanas indexadas (de ~151).")
    print(f"{'Objeto':<18} {'archivos':>9} {'épocas':>7} {'recortes':>9} {'resultado':>10}")
    for n in NAMES:
        ra, dec = source.resolve_name(n)
        files = query._from_index_offline(ra, dec)
        t = sorted(f["t"] for f in files)
        ep = 1 + sum(b - a > 30 for a, b in zip(t, t[1:])) if t else 0
        carpeta = source.ROOT / "cache" / f"{ra:.4f}_{dec:.4f}_{FOV}"
        nrec = len(list(carpeta.glob("*.fits"))) if carpeta.exists() else 0
        print(f"{n:<18} {len(files):>9} {ep:>7} {nrec:>9} {'sí' if pre_path(ra, dec, FOV).exists() else 'no':>10}")


def run(targets, force=False):
    import api          # importa la ciencia una sola vez

    source.SOURCE = "irsa"      # datos reales
    hechos, fallos = [], []
    for name, ra, dec in targets:
        paso = "1. resolver el nombre"
        t0 = time.time()
        try:
            if ra is None:
                ra, dec = source.resolve_name(name)
            out = api.pre_path(ra, dec, FOV)
            print(f"\n=== {name}  RA={ra:.4f} Dec={dec:.4f}")
            if out.exists() and not force:
                print("  ya estaba preparado (usa --force para recalcular)")
                hechos.append((name, "ya estaba"))
                continue
            paso = "2-4. buscar, bajar y procesar"
            res = api.compute(ra, dec, FOV)
            paso = "5. guardar"
            api.PRE.mkdir(exist_ok=True)
            out.write_text(json.dumps(res))
            a = res.get("astro") or {}
            sh = a.get("shift")
            resumen = (f"{res['n_exposures']} exposiciones, {len(res['epochs'])} época(s)"
                       + (f", movimiento {sh['total']:.2f}\"" if sh else "")
                       + (", SATURADA" if res.get("saturated") else ""))
            print(f"  listo en {time.time() - t0:.0f} s: {resumen}")
            hechos.append((name, resumen))
        except Exception as err:
            print(f"  FALLÓ en el paso {paso}: {type(err).__name__}: {err}")
            fallos.append((name, paso, str(err)[:120]))

    print("\n" + "=" * 60)
    for n, r in hechos:
        print(f"  OK     {n}: {r}")
    for n, p, e in fallos:
        print(f"  FALLÓ  {n} en {p}: {e}")
    print(f"{len(hechos)} listos, {len(fallos)} con problemas.")


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--status" in args:
        status()
    else:
        run(targets_from_args([a for a in args if a != "--force"]), force="--force" in args)
