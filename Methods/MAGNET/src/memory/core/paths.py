from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path


def _resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


_REPO_ROOT = _resolve_repo_root()
_SRC_ROOT = Path(__file__).resolve().parents[1]


def repo_root() -> Path:
    """Return the absolute path to the project root."""
    return _REPO_ROOT


def package_root() -> Path:
    """Return the path that contains the memory package."""
    return _SRC_ROOT


def _path_from_env(var_name: str, default: Path) -> Path:
    value = os.environ.get(var_name)
    if value:
        return Path(value).expanduser().resolve()
    return default


def cache_root() -> Path:
    default = repo_root() / "artifacts" / "runtime"
    return _path_from_env("GUI_MEMORY_CACHE_ROOT", default)


def log_dir() -> Path:
    default = repo_root() / "logs"
    return _path_from_env("GUI_MEMORY_LOG_DIR", default)


def data_root() -> Path:
    default = repo_root() / "data"
    return _path_from_env("GUI_MEMORY_DATA_ROOT", default)


def storage_root() -> Path:
    default = repo_root() / "artifacts" / "storage"
    return _path_from_env("GUI_MEMORY_STORAGE_ROOT", default)


def assets_dir() -> Path:
    default = repo_root() / "assets"
    return _path_from_env("GUI_MEMORY_ASSETS_ROOT", default)


def ensure_dir(path: Path | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamped_work_dir(prefix: str) -> Path:
    """Return a timestamped working directory under the cache root."""
    base = ensure_dir(cache_root())
    dirname = f"{prefix}_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    return base / dirname


def log_path(filename: str) -> Path:
    return ensure_dir(log_dir()) / filename


def storage_path(*parts: str) -> Path:
    base = ensure_dir(storage_root())
    return base.joinpath(*parts)


def dataset_path(*parts: str) -> Path:
    return data_root().joinpath(*parts)


def add_repo_to_sys_path() -> None:
    repo = str(repo_root())
    if repo not in sys.path:
        sys.path.insert(0, repo)


def add_package_to_sys_path() -> None:
    pkg = str(package_root())
    if pkg not in sys.path:
        sys.path.insert(0, pkg)
