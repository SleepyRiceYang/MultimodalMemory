import argparse
from pathlib import Path

from ..config import load_config
from ..memory import build_memories


def main():
    parser = argparse.ArgumentParser(description="Build distinct frozen memories from shared historical episodes")
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    print(build_memories(args.artifacts.resolve(), load_config(args.config)))


if __name__ == "__main__":
    main()
