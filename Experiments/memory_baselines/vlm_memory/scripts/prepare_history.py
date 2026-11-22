"""Create independent, procedural smoke histories with observed model responses."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from ..config import load_config, write_json, WORKSPACE
from ..models import LocalModel
from ..prompts import MATH_SYSTEM, VISION_SYSTEM


def prepare(root: Path, config: dict) -> None:
    if (root / "history.jsonl").exists():
        raise FileExistsError("Use a fresh artifact directory; histories are immutable")
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "config.snapshot.json", config)
    assets = root / "history_assets"
    assets.mkdir(exist_ok=True)
    tasks = []
    # Procedural source tasks are independent of all benchmark examples/labels.
    image = Image.new("RGB", (420, 260), "white")
    draw = ImageDraw.Draw(image)
    for x in (30, 140, 250):
        draw.ellipse((x, 30, x + 65, 95), fill="steelblue")
    for x in (70, 220):
        draw.rectangle((x, 150, x + 65, 215), fill="coral")
    path = assets / "math_shapes.png"
    image.save(path)
    tasks.append(("mathvision", "synthetic_shapes", "Count all circles and squares in the picture. How many shapes are there?", path, "5"))
    image = Image.new("RGB", (420, 260), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((80, 60, 320, 220), outline="black", width=3)
    draw.text((180, 35), "4 cm", fill="black")
    draw.text((325, 130), "3 cm", fill="black")
    path = assets / "math_rectangle.png"
    image.save(path)
    tasks.append(("mathvision", "synthetic_rectangle", "The rectangle is 4 cm wide and 3 cm high. What is its area in square cm?", path, "12"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for index, horizontal in enumerate((False, True)):
        fig, ax = plt.subplots(figsize=(5.12, 5.12), dpi=100)
        if horizontal:
            ax.barh(["D", "E", "F"], [4, 1, 6], color="teal")
        else:
            ax.bar(["A", "B", "C"], [2, 5, 3], color="steelblue")
        ax.set_title("Independent synthetic bar chart")
        fig.tight_layout()
        path = assets / f"chart_{index}.png"
        fig.savefig(path)
        plt.close(fig)
        tasks.append(("vision2code", f"synthetic_chart_{index}",
                      "Recreate the reference bar chart using matplotlib and save the figure to OUTPUT_PATH.", path, None))
    model = LocalModel(config["model"])
    episodes = []
    sys.path.insert(0, str(WORKSPACE / "Benchmark/Vision2Code"))
    from vision2code.generation.normalize_code import normalize_generated_code
    from vision2code.rendering.render_python import render_matplotlib_code
    for benchmark, source_id, task, path, expected in tasks:
        print(f"Collecting independent history: {source_id}", flush=True)
        response = model.generate(MATH_SYSTEM if benchmark == "mathvision" else VISION_SYSTEM,
                                   task, [("Historical task reference", path)])
        write_json(root / "history_calls" / f"{source_id}.json", response)
        images = [{"path": str(path.relative_to(root)), "role": "reference_before_action"}]
        steps = [{"kind": "observation", "text": task, "image_index": 0},
                 {"kind": "model_response", "text": response["text"]}]
        if benchmark == "mathvision":
            boxes = re.findall(r"\\boxed\{([^{}]+)\}", response["text"])
            extracted = boxes[-1].strip() if boxes else ""
            success = extracted == expected
            feedback = {"expected": expected, "model_answer": extracted, "correct": success,
                        "origin": "procedural_generator_known_answer"}
        else:
            output = assets / f"{source_id}_render.png"
            render = render_matplotlib_code(normalize_generated_code(response["text"]), output)
            success = bool(render["render_success"])
            feedback = {"render_success": success, "status": render["status"],
                        "semantic_fidelity": "not_judged", "success_definition": "renderability_only"}
            if success:
                images.append({"path": str(output.relative_to(root)), "role": "render_after_action"})
                steps.append({"kind": "render", "text": "Python executed and produced an image.", "image_index": 1})
        episodes.append({"id": source_id, "benchmark": benchmark, "task": task,
                         "provenance": "independent_procedural_smoke_fixture_v1",
                         "images": images, "steps": steps, "feedback": feedback, "success": success})
    (root / "history.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in episodes))
    write_json(root / "history_manifest.json", {"source_count": len(episodes), "benchmark_examples_used": 0,
               "purpose": "pipeline validation only; not a research-quality memory corpus"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    prepare(args.artifacts.resolve(), load_config(args.config))


if __name__ == "__main__":
    main()
