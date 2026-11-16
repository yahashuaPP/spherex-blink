# 🔭 SPHEREx Blink — NASA Space Apps 2026 (reto "Planet X and SPHEREx")

**Equipo:** Yahashua (Física + Computación) · Ezequiel Guerrero (Física + Matemática)
**Sede:** Quito (EPN), participación virtual
**Fechas:** sábado 14 y domingo 15 de noviembre de 2026
**Meta:** **entregar el domingo a las 17:00** como máximo. Confirmar con la sede la hora límite real y no apurarse al final.

---

## 1. Qué construimos (en una frase)

Una web donde el usuario escribe **RA/Dec** (o un nombre como "Barnard's Star") y ve el pedazo de cielo de SPHEREx **animado por épocas**, con **imagen diferencia**, **campo de visión ajustable**, **Gaia encima** y, como extra, el **espectro** de la fuente.

### Demo mínima que TIENE que salir (MVP)
- [ ] Entrar RA/Dec y ver un GIF o animación con un cuadro por época.
- [ ] Funciona con los 5 objetos de prueba (ver §6).
- [ ] Coordenadas en la URL: `?ra=269.45&dec=4.69&fov=0.1`.

### Extras (en este orden, solo si sobra tiempo)
1. Imagen diferencia (época N menos época 1).
2. Aladin Lite al lado con Gaia DR3.
3. Co-add por detector (6 detectores) para mejor señal/ruido.
4. Espectro de la fuente al vuelo ("extra kudos" según el reto).
5. Mostrar los flags de contaminación.

---

## 2. Arquitectura

```
[Navegador]  index.html + JS (blink, slider FOV, botón diferencia, Aladin Lite)
     │  GET /frames?ra=..&dec=..&fov=..
     ▼
[Backend FastAPI]  (YAHASHUA)
     ├─ query.py     → busca en IRSA las exposiciones que cubren (ra, dec)
     ├─ download.py  → baja los recortes con cola de prioridad + caché en disco
     └─ api.py       → llama a science.py y devuelve PNGs + metadata (JSON)
     ▼
[Ciencia]  science.py  (EZEQUIEL)
     ├─ reproyectar a una grilla común
     ├─ agrupar por época → co-add (pesado por 1/varianza, con máscara de flags)
     ├─ diferencias entre épocas
     └─ espectro (fotometría de apertura por exposición, con su λ)
```

### Cola de prioridad (la idea de Yahashua)
- Por cada posición hay **cientos** de exposiciones. No esperamos a tenerlas todas.
- Prioridad = `(n_ya_bajadas_de_esa_época, detector)`. El menor va primero, así se **intercalan las épocas**: rápido hay al menos 1 imagen por época y el video ya se puede mostrar. Luego se rellena el resto.
- Hasta **8 descargas en paralelo** (asyncio + httpx). Tope de unas **30 exposiciones por época**.
- Caché en disco: `cache/{ra:.4f}_{dec:.4f}_{fov}/archivo.fits`. La segunda vez carga al instante.

---

## 3. 🤝 El CONTRATO entre los dos (lo más importante)

Si los dos respetan esto, trabajan en paralelo sin depender uno del otro.

**Yahashua entrega** (salida de `download.py`):
```python
exposures: list[dict] = [
    {
        "t": float,              # MJD de la observación
        "detector": int,         # 1..6
        "image": np.ndarray,     # HDU "IMAGE"
        "variance": np.ndarray,  # HDU "VARIANCE"
        "flags": np.ndarray,     # HDU "FLAGS"
        "wcs": astropy.wcs.WCS,  # WCS espacial del IMAGE
        "path": str,             # ruta del FITS (para leer WCS-WAVE si hace falta)
    },
    ...
]
```

**Ezequiel entrega** (salida de `science.py`):
```python
def build_frames(exposures, ra, dec, fov) -> dict:
    return {
        "epochs":   [{"t_mean": float, "image": np.ndarray, "n_exp": int}, ...],
        "diff":     np.ndarray | None,                    # última época − primera
        "spectrum": {"lam_um": np.ndarray, "flux": np.ndarray, "err": np.ndarray} | None,
        "wcs":      astropy.wcs.WCS,                      # grilla común
    }
```

**Regla:** mientras el descargador no esté listo, Ezequiel trabaja con 5–10 FITS de Barnard bajados a mano, en `data/sample/`.

---

## 4. 👤 Tareas de YAHASHUA (datos + backend + web)

| # | Tarea | Listo cuando… |
|---|-------|----------------|
| Y1 | `query.py`: dado (ra, dec), lista de URLs de recorte + tiempo + detector | imprime N exposiciones de Barnard ordenadas por tiempo |
| Y2 | `download.py`: cola de prioridad + async + caché | baja 60 recortes de Barnard sin errores y la 2.ª vez no baja nada |
| Y3 | Leer cada FITS → dict del contrato | `exposures[0]` tiene todos los campos |
| Y4 | `api.py` con FastAPI: `/frames` y `/resolve?name=` | `curl localhost:8000/frames?ra=..` devuelve JSON con PNGs en base64 |
| Y5 | `index.html`: inputs, blink animado (play/pausa, velocidad), slider FOV, lectura de `?ra&dec&fov` de la URL | se ve Barnard animado en el navegador |
| Y6 | Aladin Lite + capa Gaia DR3 | aparece el mismo campo con Gaia al lado |
| Y7 | Precalcular los 5 objetos de prueba (la demo no depende de IRSA en vivo) | `cache/` lleno, demo instantánea |
| Y8 | Repo, README final, deploy si da (Render o Hugging Face Spaces) | link público funcionando |

### Esqueleto para empezar (Y1)
```python
# query.py
import re
import pyvo

SIA = pyvo.dal.SIA2Service("https://irsa.ipac.caltech.edu/SIA")

def find_exposures(ra, dec, fov_deg=0.1, max_n=400):
    res = SIA.search(pos=(ra, dec, 0.001), collection="spherex_qr")  # revisar si hay colección más nueva (ej. qr2)
    tab = res.to_table()
    out = []
    for row in tab[:max_n]:
        url = str(row["access_url"])
        m = re.search(r"_2D(\d)_", url)          # el detector viene en el nombre del archivo
        out.append({
            "url": f"{url}?center={ra},{dec}&size={fov_deg}",
            "t": float(row["t_min"]),
            "detector": int(m.group(1)) if m else -1,
        })
    return sorted(out, key=lambda d: d["t"])
```
> Si SIA2 falla o es lento, copiar el query TAP del tutorial oficial de PSF/cutouts de IRSA (tablas `spherex.artifact` y `spherex.plane`), que ya devuelve las URLs de recorte con su tiempo.

---

## 5. 👤 Tareas de EZEQUIEL (ciencia de las imágenes)

| # | Tarea | Listo cuando… |
|---|-------|----------------|
| C1 | Bajar a mano 5–10 recortes de Barnard y leer los HDUs (IMAGE, FLAGS, VARIANCE, WCS-WAVE) | entiende la estructura y las unidades |
| C2 | **Corregir coordenadas por movimiento propio** a la fecha de SPHEREx (ver §6, ¡crítico!) | la estrella cae en el centro del recorte |
| C3 | Reproyectar todo a una grilla común (`reproject.reproject_interp`) | dos exposiciones de la misma época se superponen bien |
| C4 | Agrupar por época (clustering de tiempos: saltos > 30 días = época nueva) | Barnard da ~3 grupos |
| C5 | Co-add por época: media pesada por 1/varianza, píxeles con flags malos enmascarados | imagen por época limpia |
| C6 | Imagen diferencia (normalizada, colormap divergente) | Barnard se ve como par blanco/negro |
| C7 | **Validación:** medir el desplazamiento del centroide entre épocas y compararlo con el movimiento propio de Gaia | tabla "medido vs esperado" para el pitch |
| C8 | Espectro: fotometría de apertura en cada exposición, λ del pixel central vía WCS-WAVE | gráfico λ vs flujo de Barnard |
| C9 | Escalado de display (asinh o zscale) para que los PNG se vean bien | las imágenes no salen ni negras ni quemadas |

---

## 6. 🎯 Objetos de prueba

| Objeto | Qué se debe ver | Nota |
|--------|-----------------|------|
| Barnard's Star | movimiento grande (~10″/año) | |
| Epsilon Indi B (Ba/Bb) | par de enanas marrones moviéndose (~4.7″/año) | |
| WISE 0855 (RA 08h55m10.831s, Dec −07°14′42.53″, coordenadas que da el reto) | enana marrón rápida (~8″/año), muy débil | puede salir muy tenue: es normal y se puede decir en el pitch |
| RW Aurigae | **cambio de brillo**, no de posición | |
| R Draconis | variable Mira: cambio de brillo grande | |

⚠️ **CRÍTICO:** SIMBAD da coordenadas en la época J2000. Barnard se movió **~4′** desde 2000. Si usamos la coordenada J2000 tal cual, el recorte sale corrido.
**Solución:** `Simbad` con `pmra/pmdec`, luego `SkyCoord(..., obstime="J2000").apply_space_motion(new_obstime=Time("2025-11-01"))`. Para WISE 0855 hay que revisar a qué época corresponden las coordenadas del reto y corregir igual.
Los píxeles de SPHEREx son de ~6″, así que el movimiento entre la primera y la última época es de solo **1–3 píxeles**. La alineación tiene que ser buena.

---

## 7. ⏰ Cronograma (hora de Quito)

### Sábado 14
| Hora | Yahashua | Ezequiel | Juntos |
|------|----------|-----------|--------|
| 09:00–09:30 | | | ☕ **Kickoff:** leer este README, crear el repo, acordar el contrato (§3) |
| 09:30–12:30 | Y1, Y2 | C1, C2, C3 | |
| 12:30–13:00 | | | 🔗 **Sync 1:** ¿sale Barnard de IRSA? ¿la grilla común funciona? |
| 13:00–14:00 | 🍽️ almuerzo | 🍽️ almuerzo | |
| 14:00–18:00 | Y3, Y4 | C4, C5, C6 | |
| 18:00–19:00 | | | 🔗 **INTEGRACIÓN 1:** `download.py` + `science.py` → un GIF de Barnard por épocas. **Aunque sea feo.** |
| 19:00–22:00 | Y5 (frontend) | C7 (validación del movimiento) | |
| 22:00 | | | 🔗 **Sync 2:** la web muestra Barnard animado. A dormir. 😴 |

### Domingo 15
| Hora | Yahashua | Ezequiel | Juntos |
|------|----------|-----------|--------|
| 08:30–11:30 | Y6 (Aladin + Gaia), Y7 (precalcular los 5 objetos) | C8 (espectro), C9 | |
| 11:30–12:30 | | | 🔗 **INTEGRACIÓN 2:** los 5 objetos + diferencia + espectro en la web |
| 12:30–13:00 | 🍽️ | 🍽️ | |
| 13:00–14:30 | Y8 (deploy o repo limpio) | Slides y textos de la página del proyecto | |
| 14:30–15:30 | | | 🎬 **Video demo** (≤ 30 s de pantalla + voz) |
| 15:30–17:00 | | | 📤 **Entrega** en la web de Space Apps + revisión final |

**🔒 Congelamiento:** después de las **14:30 del domingo no se agregan features**. Solo se arreglan bugs.

---

## 8. 🧯 Plan B (si algo se rompe)

| Problema | Plan B |
|----------|--------|
| IRSA lento o caído | usar la caché precalculada (Y7). Por eso se hace temprano el domingo. |
| La reproyección no queda bien | alinear con el WCS más cercano y recortar al mismo tamaño en píxeles. |
| FastAPI o el deploy dan problemas | Streamlit en una sola app y la demo local en el video. |
| El espectro no sale | se presenta como "trabajo futuro" con un gráfico de prueba de Barnard. |
| WISE 0855 no se ve | se explica: es muy débil en el NIR y se muestra la posición esperada sobre la imagen. |

---

## 9. 🛠️ Setup (hacerlo los dos HOY antes de dormir o al despertar)

```bash
python -m venv venv && source venv/bin/activate      # en Windows: venv\Scripts\activate
pip install astropy astroquery pyvo reproject numpy matplotlib httpx fastapi uvicorn pillow
git clone <repo> && cd <repo>
```

Estructura del repo:
```
spherex-blink/
├── backend/   query.py  download.py  api.py      ← Yahashua
├── science/   science.py  notebooks/              ← Ezequiel
├── web/       index.html  app.js  style.css       ← Yahashua
├── data/sample/   cache/                          ← (cache en .gitignore)
└── README.md
```

**Git:** cada uno trabaja en su rama (`yahashua`, `ciencia`) y se hace merge **solo en las integraciones**. Commits chicos y seguidos.

---

## 10. 📤 Checklist de entrega

- [ ] Título y resumen corto del proyecto
- [ ] Link al repo (público)
- [ ] Link a la demo (deploy) o video
- [ ] Video demo o slides
- [ ] Datos de NASA usados: **SPHEREx (IRSA)**, **Gaia DR3**, y mencionar WiseView como inspiración
- [ ] Declarar el uso de IA si las reglas lo piden
- [ ] Los dos miembros del equipo registrados en el proyecto
- [ ] Confirmar con la sede de Quito cómo es la presentación virtual

---

## 11. 🎤 Pitch en 4 frases
1. Plutón se descubrió comparando dos fotos del cielo. SPHEREx toma esas fotos cada 6 meses, en 102 colores.
2. Nuestra herramienta convierte el archivo de SPHEREx en un "blink" que cualquiera puede usar: escribes una coordenada y ves qué se mueve.
3. Lo validamos midiendo el movimiento de Barnard's Star y comparándolo con Gaia, y además sacamos su espectro.
4. Es abierta, embebible por URL y se actualiza sola con cada liberación nueva de datos.

---

## 12. ▶️ Cómo correr

```bash
pip install -r requirements.txt
cd backend
uvicorn api:app --reload
```

Abrir http://localhost:8000 y buscar `Barnard's Star`.

Por defecto usa **datos sintéticos** (`make_fake.py`). Para usar datos reales de IRSA y SIMBAD:

```powershell
$env:SPHEREX_SOURCE = "irsa"      # PowerShell
```
```bash
export SPHEREX_SOURCE=irsa        # bash
```
