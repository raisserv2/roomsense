# Build context: the repo root (see deploy/compose.yaml and /.dockerignore)
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HOME=/tmp
WORKDIR /srv
COPY deploy/api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY roomsense/ roomsense/
COPY deploy/api/app/ app/
USER 1000:1000
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
