#!/usr/bin/env bash
# Isolated AVDs; share only the base emulator's read-only APK/cache/server mounts.
# Only starts environments, never tasks or model requests.
set -euo pipefail
BASE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
WORKERS="${1:-4}"
if [[ ! "${WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
  echo 'Expected a positive worker count' >&2
  exit 1
fi
ANDROID_CONTAINER=magnet_cold_aw ANDROID_PORT=15006 \
  bash "${BASE}/Experiments/memory_baselines/scripts/experiments/prepare_androidworld_magnet_cold.sh"
for ((i=1; i<WORKERS; i++)); do
  name="magnet_cold_aw_${i}"
  if ! docker container inspect "${name}" >/dev/null 2>&1; then
    docker create --name "${name}" --device /dev/kvm --cap-add NET_ADMIN \
      --sysctl net.ipv6.conf.all.disable_ipv6=0 \
      --sysctl net.ipv6.conf.default.disable_ipv6=0 \
      --sysctl net.ipv6.conf.eth0.disable_ipv6=0 \
      --volumes-from magnet_cold_aw:ro \
      -e ANDROID_WORLD_A11Y_APK=/opt/android_world/cache/2024.05.13-accessibility_forwarder.apk \
      -p "127.0.0.1:$((15006+i)):5000" android_world:latest
  fi
  if [[ "$(docker inspect -f '{{.State.Running}}' "${name}")" != true ]]; then docker start "${name}"; fi
  printf 'Worker %s: %s http://127.0.0.1:%s/health\n' "${i}" "${name}" "$((15006+i))"
done
