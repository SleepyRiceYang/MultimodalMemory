# MAGNET: Towards Adaptive GUI Agents with Memory-Driven Knowledge Evolution

This repository hosts materials for the paper "MAGNET: Towards Adaptive GUI Agents with Memory-Driven Knowledge Evolution".

- Paper: https://arxiv.org/abs/2601.19199

## Quick Run

```bash
PYTHONPATH=src python scripts/run_agent.py --config configs/presets/coat/qwenvl_android.yaml \
  --stage flow --limit 1 --dataset-dir sample_data/android_in_the_zoo --split test
```

```bash
GUI_MEMORY_STORAGE_ROOT=/remote-home/lbsun/Magnet/artifacts/storage PYTHONPATH=src \
python scripts/run_agent.py --config configs/presets/coat/qwenvl_android_hint.yaml \
  --stage all --memory-mode stationary --limit 1 --dataset-dir sample_data/android_in_the_zoo --split test
```
