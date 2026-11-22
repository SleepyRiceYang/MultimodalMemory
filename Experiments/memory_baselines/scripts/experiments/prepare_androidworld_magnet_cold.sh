#!/usr/bin/env bash
# Start an isolated emulator. Does not restart/modify the existing vLLM service.
set -euo pipefail
BASE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
CONTAINER="${ANDROID_CONTAINER:-magnet_cold_aw}"
PORT="${ANDROID_PORT:-15006}"
# Bind paths are interpreted by the Docker daemon, not by this shell.
HOST_BASE="${DOCKER_WORKSPACE:-${BASE}}"
if ! docker container inspect "${CONTAINER}" >/dev/null 2>&1; then
  docker create --name "${CONTAINER}" --device /dev/kvm --cap-add NET_ADMIN \
    --sysctl net.ipv6.conf.all.disable_ipv6=0 \
    --sysctl net.ipv6.conf.default.disable_ipv6=0 \
    --sysctl net.ipv6.conf.eth0.disable_ipv6=0 \
    -e ANDROID_WORLD_A11Y_APK=/opt/android_world/cache/2024.05.13-accessibility_forwarder.apk \
    -p "127.0.0.1:${PORT}:5000" \
    --mount "type=bind,src=${HOST_BASE}/Benchmark/AndroidWorld/.cache/android_env,dst=/opt/android_world/cache,readonly" \
    --mount "type=bind,src=${HOST_BASE}/Benchmark/AndroidWorld/.cache/android_world/app_data,dst=/tmp/android_world/app_data,readonly" \
    --mount "type=bind,src=${HOST_BASE}/Benchmark/AndroidWorld/server/android_server.py,dst=/app/server/android_server.py,readonly" \
    --mount "type=bind,src=${HOST_BASE}/Benchmark/AndroidWorld/docker_setup/start_emu_headless.sh,dst=/app/docker_setup/start_emu_headless.sh,readonly" \
    android_world:latest
fi
if [[ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER}")" != true ]]; then
  docker start "${CONTAINER}"
fi
printf 'AndroidWorld: http://127.0.0.1:%s/health\n' "${PORT}"
printf 'Check startup with: docker logs --tail 30 %s\n' "${CONTAINER}"
