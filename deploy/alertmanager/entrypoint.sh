#!/bin/sh
# Builds alertmanager.yml from environment variables, then starts Alertmanager.
# Telegram is optional: without TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID only the
# retrain webhook is configured.
set -e
CFG=/alertmanager/alertmanager.yml

telegram() {
  if [ -n "$TELEGRAM_BOT_TOKEN" ] && [ -n "$TELEGRAM_CHAT_ID" ]; then
    cat <<YAML
    telegram_configs:
      - bot_token: '$TELEGRAM_BOT_TOKEN'
        chat_id: $TELEGRAM_CHAT_ID
        send_resolved: true
        parse_mode: ''
        message: '{{ if eq .Status "firing" }}🔴{{ else }}🟢{{ end }} {{ .CommonLabels.alertname }}: {{ .CommonAnnotations.summary }}'
YAML
  fi
}

{
  cat <<YAML
route:
  receiver: notify
  group_by: [alertname]
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 4h
  routes:
    - receiver: retrain
      matchers: ['action="retrain"']
receivers:
  - name: notify
YAML
  telegram
  cat <<YAML
  - name: retrain
    webhook_configs:
      - url: http://api:8000/hooks/alertmanager
        send_resolved: true
YAML
  telegram
} > "$CFG"

exec /bin/alertmanager --config.file="$CFG" --storage.path=/alertmanager "$@"
