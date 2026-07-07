#!/usr/bin/env bash
# Validate Caddy with the same EnvironmentFile values systemd injects.
# This avoids shell-expanding bcrypt hashes or shared keys containing "$".
set -euo pipefail

env_file="${1:-/etc/default/caddy}"
config_file="${2:-/etc/caddy/Caddyfile}"

if [ ! -r "$env_file" ]; then
    echo "Cannot read $env_file" >&2
    exit 1
fi

while IFS= read -r line; do
    case "$line" in
        ""|\#*) continue ;;
        *=*)
            key="${line%%=*}"
            value="${line#*=}"
            case "$key" in
                *[!A-Za-z0-9_]*|"") continue ;;
            esac
            export "$key=$value"
            ;;
    esac
done < "$env_file"

exec caddy validate --config "$config_file"
