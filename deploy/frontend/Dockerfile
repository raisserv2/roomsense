FROM nginx:1.27-alpine
# Defaults; override in deploy/.env to point the UI at another backend.
ENV API_UPSTREAM=http://api:8000 API_BASE_URL=api/
# The image renders /etc/nginx/templates/*.template with envsubst at start-up.
COPY default.conf.template /etc/nginx/templates/default.conf.template
COPY 40-roomsense-config.sh /docker-entrypoint.d/40-roomsense-config.sh
RUN chmod +x /docker-entrypoint.d/40-roomsense-config.sh
COPY site/ /usr/share/nginx/html/
EXPOSE 80
