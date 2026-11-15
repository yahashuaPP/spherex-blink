# Demo pública de SPHEREx Blink: sirve la web y los resultados ya calculados (precomputed/).
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backend backend
COPY science science
COPY web web
COPY precomputed precomputed
# demo = solo objetos preparados (sin descargas ni cálculo en el servidor)
ENV SPHEREX_SOURCE=demo MPLCONFIGDIR=/tmp/mpl
WORKDIR /app/backend
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]
