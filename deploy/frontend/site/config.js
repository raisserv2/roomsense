// Default settings for opening the site without Docker.
// In the container this file is regenerated at start-up from environment
// variables (deploy/frontend/40-roomsense-config.sh), so the API address is
// configuration and never hard-coded in the pages.
window.ROOMSENSE_CONFIG = {
  apiBase: "api/",
  links: {"Capture": "7443", "MLflow": "5443", "Grafana": "3443", "Prometheus": "6443"}
};
