#!/usr/bin/env python3
"""Install MMSkills runtime additively into an OSWorld checkout."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
MODULES = (
    "general_agent.py",
    "general_skill_agent.py",
    "general_text_skill_agent.py",
    "mm_skill_agent.py",
    "skill_loader.py",
    "task_skill_resolver.py",
)
OLD_IMPORT = "from mm_agents.utils.qwen_vl_utils import preprocess_image_for_vlm"
NEW_IMPORT = "from mm_agents.utils.mmskills_qwen_vl_utils import preprocess_image_for_vlm"


def installed_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").replace(OLD_IMPORT, NEW_IMPORT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("osworld_root", type=Path)
    args = parser.parse_args()
    osworld = args.osworld_root.expanduser().resolve()
    agents = osworld / "mm_agents"
    if not (osworld / "desktop_env" / "desktop_env.py").is_file():
        raise SystemExit(f"Not an OSWorld checkout: {osworld}")
    agents.mkdir(parents=True, exist_ok=True)

    # agent.py belongs to OSWorld and supplies parse_code_from_string; never replace it.
    if not (agents / "agent.py").is_file():
        raise SystemExit(f"Expected OSWorld's existing {agents / 'agent.py'}")
    for name in MODULES:
        src, dst = REPO / "mm_agents" / name, agents / name
        if dst.exists() and dst.read_text(encoding="utf-8") != installed_text(src):
            raise SystemExit(f"Refusing to overwrite existing file: {dst}")
        if not dst.exists():
            shutil.copy2(src, dst)
            dst.write_text(installed_text(src), encoding="utf-8")

    internal_src = REPO / "mm_agents" / "_mmskills_internal"
    internal_dst = agents / "_mmskills_internal"
    for src in internal_src.rglob("*"):
        if not src.is_file() or "__pycache__" in src.parts:
            continue
        dst = internal_dst / src.relative_to(internal_src)
        if dst.exists() and dst.read_bytes() != src.read_bytes():
            raise SystemExit(f"Refusing to overwrite existing file: {dst}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(src, dst)

    # OSWorld has its own qwen_vl_utils.py; MMSkills requires an extra helper absent there.
    src = REPO / "mm_agents" / "utils" / "qwen_vl_utils.py"
    dst = agents / "utils" / "mmskills_qwen_vl_utils.py"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.read_bytes() != src.read_bytes():
        raise SystemExit(f"Refusing to overwrite existing file: {dst}")
    if not dst.exists():
        shutil.copy2(src, dst)
    print(f"MMSkills runtime installed additively into {osworld}")
    print(f"Preserved: {agents / 'agent.py'} and {agents / 'utils/qwen_vl_utils.py'}")


if __name__ == "__main__":
    main()
