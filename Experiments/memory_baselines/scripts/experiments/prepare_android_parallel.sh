#!/usr/bin/env bash
# Creates two independent emulators; shares only read-only APK/cache/server mounts.
set -euo pipefail
BASE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
for i in 0 1; do
  name="gui_async_aw_${i}"
  if ! docker container inspect "$name" >/dev/null 2>&1; then
    docker create --name "$name" --device /dev/kvm --cap-add NET_ADMIN \
      --sysctl net.ipv6.conf.all.disable_ipv6=0 \
      --sysctl net.ipv6.conf.default.disable_ipv6=0 \
      --sysctl net.ipv6.conf.eth0.disable_ipv6=0 \
      --volumes-from android_world_smoke:ro \
      -e ANDROID_WORLD_A11Y_APK=/opt/android_world/cache/2024.05.13-accessibility_forwarder.apk \
      -p "127.0.0.1:$((15003+i)):5000" android_world:latest
    # start_emu_headless.sh is already shared read-only from android_world_smoke
    # by --volumes-from above, so docker cp cannot replace it in the child.
    docker cp "$BASE/Benchmark/AndroidWorld/android_world/task_evals/single/sms.py" "$name:/app/android_world/task_evals/single/sms.py"
  fi
  if [[ "$(docker inspect -f '{{.State.Running}}' "$name")" != true ]]; then docker start "$name"; fi
done
