# ---------- 1. Etap build (lżejszy) ----------
FROM python:3.11-slim AS base

#    optymalizacje: brak cache PIP, ustaw locale
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

# ---------- 2. Instalacja zależności ----------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------- 3. Kopiujemy kod i modele ----------
COPY venv/src       src/
COPY venv/models    models/
# (scaler.pkl + *.pkl + nn.h5)

# ---------- 4. Komenda startowa ----------
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
