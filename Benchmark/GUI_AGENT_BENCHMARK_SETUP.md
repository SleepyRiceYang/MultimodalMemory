# GUI Agent Benchmark 配置记录

> 截至 2026-10-21；记录当前仓库中的 AndroidWorld 与 OSWorld 本地配置和已完成的验证，不代表完整基准测试成绩。

## 共用运行条件

两个代码仓库分别位于 `Benchmark/AndroidWorld/` 和 `Benchmark/OSWorld/`。当前实验容器通过挂载宿主机 Docker socket 调用宿主机 Docker，并可访问 `/dev/kvm`；无需在实验容器内启动 `dockerd`。Docker socket 赋予近似宿主机 root 的控制能力，只操作本项目创建的镜像与容器，不清理管理员的资源。宿主机与实验容器共享的工作目录分别是 `/data/yangshengjie/workspace/VLM-Memory` 与 `/root/VLM-Memory`；Docker bind mount 应使用**宿主机路径**。网络下载不稳定时，已有脚本可使用 `127.0.0.1:13876` 代理，首次获取资源仍可能需要联网。

## AndroidWorld

- 镜像：`android_world:latest`；命名容器：`android_world_smoke`。使用 Android Emulator（Android 13 / API 33）、`/dev/kvm` 和本地端口 `127.0.0.1:15000`。
- 模拟器、应用数据与 accessibility-forwarder APK 已缓存；相关脚本位于 `docker_setup/`，包括 `download_android_emulator.sh`、`prefetch_app_data.sh` 和 `start_android_world.sh`。重启现有容器通常无需重新下载。
- 启动与检查：

  ```bash
  bash Benchmark/AndroidWorld/docker_setup/start_android_world.sh
  curl --noproxy '*' http://127.0.0.1:15000/health
  docker exec android_world_smoke adb shell getprop sys.boot_completed
  ```

  健康检查应返回 `status: success`，启动完成属性应为 `1`。已验证重启后模拟器正常启动，并完成过任务初始化、评分与清理流程；这不是 Agent 成功率测试。需要停止时运行 `docker stop android_world_smoke`。

## OSWorld

- Python 环境：Conda `osworld`（评测客户端）与 `vllm_env`（模型服务）。Docker 镜像：`happysixd/osworld-docker:latest`。桌面 VM 文件为 `Benchmark/OSWorld/docker_vm_data/Ubuntu.qcow2`，由同目录 `scripts/setup_local_docker_ubuntu.sh` 下载、校验并解压；已完成下载。
- `scripts/quickstart_local_docker_ubuntu.sh` 已通过环境重置、动作执行和关闭测试。正式任务由 Docker 后端临时创建 QEMU/KVM 容器，而非复用 AndroidWorld 的命名容器。
- 单任务模型验证使用本地 `/data/pretrained_models/Qwen3-VL-8B-Instruct`，默认 GPU 0，执行：

  ```bash
  cd Benchmark/OSWorld
  bash scripts/run_local_qwen_smoke.sh
  # 快速重启验证可用：OSWORLD_MAX_STEPS=1 bash scripts/run_local_qwen_smoke.sh
  ```

  脚本检查显存和端口，启动 vLLM，运行一个 Chrome 任务，退出时停止模型服务。已完成三步模型测试和脚本重启测试，任务分数均为 `0.0`：环境与评测链路可运行，但模型没有完成该任务，不能视为基准成绩。结果保存在 `results/local_qwen_smoke/` 等目录。历史测试结束后已确认无残留 vLLM 服务或 OSWorld 容器；**再次运行前仍须检查 GPU 0 空闲显存**（脚本要求至少约 54 GiB），勿停止其他用户的进程。

## 尚未覆盖

两项均未完成全量任务评测。AndroidWorld 的某些应用/任务可能需要联网；OSWorld 的部分任务需要额外资产、网络服务或账号。现有验证只覆盖上述本地启动与单任务流程，不保证全部任务开箱即用。
