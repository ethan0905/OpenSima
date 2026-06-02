#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVER_DIR="${MINECRAFT_SERVER_DIR:-"$ROOT_DIR/minecraft-server"}"
MANIFEST_URL="https://launchermeta.mojang.com/mc/game/version_manifest.json"
MEMORY_MIN="${MINECRAFT_SERVER_XMS:-1G}"
MEMORY_MAX="${MINECRAFT_SERVER_XMX:-2G}"
PORT="${MINECRAFT_PORT:-25565}"
ONLINE_MODE="${MINECRAFT_ONLINE_MODE:-false}"
REQUESTED_VERSION="${MINECRAFT_VERSION:-latest}"
GAMEMODE="${MINECRAFT_GAMEMODE:-survival}"
DIFFICULTY="${MINECRAFT_DIFFICULTY:-easy}"

mkdir -p "$SERVER_DIR"

MANIFEST_FILE="$SERVER_DIR/version_manifest.json"
VERSION_FILE="$SERVER_DIR/version.json"

echo "[minecraft-server] Directory: $SERVER_DIR"

if ! command -v java >/dev/null 2>&1; then
  echo "[minecraft-server] Java is not installed or not on PATH."
  echo "[minecraft-server] Install Java 21+ for current Minecraft releases, then rerun this script."
  exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "[minecraft-server] curl is required to download the official server jar."
  exit 1
fi

echo "[minecraft-server] Fetching official Minecraft version manifest..."
curl -fsSL "$MANIFEST_URL" -o "$MANIFEST_FILE"

read -r SELECTED_VERSION VERSION_URL SERVER_URL < <(
  python3 - "$MANIFEST_FILE" "$REQUESTED_VERSION" <<'PY'
import json
import sys
from urllib.request import urlopen

manifest_path = sys.argv[1]
requested_version = sys.argv[2]
with open(manifest_path, "r", encoding="utf-8") as handle:
    manifest = json.load(handle)

selected = manifest["latest"]["release"] if requested_version == "latest" else requested_version
version = next((item for item in manifest["versions"] if item["id"] == selected), None)
if version is None:
    raise SystemExit(f"Unknown Minecraft version: {selected}")
with urlopen(version["url"]) as response:
    version_json = json.load(response)

print(selected, version["url"], version_json["downloads"]["server"]["url"])
PY
)

JAR_FILE="$SERVER_DIR/server-$SELECTED_VERSION.jar"

echo "[minecraft-server] Selected release: $SELECTED_VERSION"
curl -fsSL "$VERSION_URL" -o "$VERSION_FILE"

if [[ ! -f "$JAR_FILE" ]]; then
  echo "[minecraft-server] Downloading server jar..."
  curl -fL "$SERVER_URL" -o "$JAR_FILE"
else
  echo "[minecraft-server] $JAR_FILE already exists; keeping existing file."
fi

cat > "$SERVER_DIR/server.properties" <<EOF
server-port=$PORT
online-mode=$ONLINE_MODE
enable-command-block=false
motd=Local AI Agent Test Server
difficulty=$DIFFICULTY
gamemode=$GAMEMODE
spawn-protection=0
EOF

if [[ "${ACCEPT_MINECRAFT_EULA:-false}" == "true" ]]; then
  echo "eula=true" > "$SERVER_DIR/eula.txt"
elif [[ ! -f "$SERVER_DIR/eula.txt" ]]; then
  echo "eula=false" > "$SERVER_DIR/eula.txt"
fi

if ! grep -q '^eula=true$' "$SERVER_DIR/eula.txt"; then
  echo "[minecraft-server] Prepared server files, but EULA is not accepted."
  echo "[minecraft-server] Review Mojang's EULA, then start with:"
  echo "  ACCEPT_MINECRAFT_EULA=true $0"
  exit 2
fi

echo "[minecraft-server] Starting server on port $PORT..."
echo "[minecraft-server] Stop it with Ctrl+C."
echo "[minecraft-server] If Java reports UnsupportedClassVersionError, install a newer Java or pin an older Minecraft server version:"
echo "  MINECRAFT_VERSION=1.21.8 ACCEPT_MINECRAFT_EULA=true $0"
cd "$SERVER_DIR"
exec java "-Xms$MEMORY_MIN" "-Xmx$MEMORY_MAX" -jar "$JAR_FILE" nogui
