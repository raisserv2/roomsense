#!/bin/sh
# Runs when the frontend container starts (the nginx image executes everything
# in /docker-entrypoint.d). Writes config.js from environment variables, so the
# API address and the tool links are settings, not something baked into the page.
set -e
cat > /usr/share/nginx/html/config.js <<CONFIG
// Generated at container start from the environment (deploy/compose.yaml).
window.ROOMSENSE_CONFIG = {
  apiBase: "${API_BASE_URL:-api/}",
  links: {
    "Capture": "${LINK_CAPTURE:-7443}",
    "MLflow": "${LINK_MLFLOW:-5443}",
    "Grafana": "${LINK_GRAFANA:-3443}",
    "Prometheus": "${LINK_PROMETHEUS:-6443}"
  }
};
CONFIG
echo "roomsense: config.js written (apiBase=${API_BASE_URL:-api/})"
