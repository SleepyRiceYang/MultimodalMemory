#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "presets" / "coat" / "config_gemini_android.yaml"

import sys

SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agent.config.loader import load_config
from datasets.datasets import build_dataset
from agent.evaluation.action import ActionEvaluator
import imagesize

DATASET_ALIASES = {
    "aitz": {"dataset": "android-in-the-zoo", "data_dir": "/remote-home/lbsun/data/android_in_the_zoo"},
    "odyssey": {"dataset": "odyssey", "data_dir": "/remote-home/lbsun/data/gui_odyssey"},
    "amex": {"dataset": "amex", "data_dir": "/remote-home/lbsun/data/amex"},
}


def _safe_copy(src: Path, dst: Path) -> None:
    """Copy helper that tolerates missing extended-attribute support."""
    try:
        shutil.copy2(src, dst)
    except OSError as exc:
        print(f"[evaluate_predictions] copy2 failed for {src} -> {dst}: {exc}; falling back to copy().")
        shutil.copy(src, dst)


def build_overrides(args: argparse.Namespace) -> Dict[str, object]:
    overrides: Dict[str, object] = {}
    data_overrides: Dict[str, object] = {}
    if args.dataset_dir:
        data_overrides["DATA_DIR"] = args.dataset_dir
    if args.split:
        data_overrides["SPLIT"] = args.split
    if args.dataset_name:
        data_overrides["DATASET"] = args.dataset_name
    if data_overrides:
        overrides["DATA"] = data_overrides
    if args.demo_mode:
        overrides["DEMO_MODE"] = args.demo_mode.upper()
    return overrides


def apply_dataset_alias(args: argparse.Namespace) -> None:
    alias = (args.dataset_name or "").lower()
    if alias in DATASET_ALIASES:
        info = DATASET_ALIASES[alias]
        args.dataset_name = info["dataset"]
        if not args.dataset_dir:
            args.dataset_dir = info.get("data_dir")


def prediction_path(step: dict, base_dir: Path) -> Path:
    subset = step["subset"]
    episode = step["episode_id"]
    step_id = step["step_id"]
    if subset == episode:
        folder = base_dir / episode
    else:
        folder = base_dir / f"{subset}-{episode}"
    return folder / f"{subset}-{episode}_{step_id}.json"


def has_prediction(payload: dict, demo_mode: str, screen_mode: str) -> bool:
    ap = payload.get("action_predict", {})
    if demo_mode not in ap:
        return False
    return screen_mode in ap[demo_mode]


def _format_action_json(action_type, detail) -> str:
    payload = {"type": action_type, "detail": detail}
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_visual_overlay(image_path: str, action_type: str | None, detail, max_display: int = 360):
    if not action_type:
        action_type = "none"
    try:
        width, height = imagesize.get(image_path)
    except Exception:
        width, height = 360, 640
    if width == 0 or height == 0:
        width, height = 360, 640
    scale = min(max_display / width, 1.0)
    display_width = round(width * scale, 2)
    display_height = round(height * scale, 2)
    markers: List[str] = []
    lower_type = action_type.lower()
    if lower_type in ("click", "long") and detail:
        if isinstance(detail, (list, tuple)) and len(detail) >= 4:
            xmin, ymin, xmax, ymax = detail[:4]
            left = xmin * scale
            top = ymin * scale
            w = max((xmax - xmin) * scale, 2)
            h = max((ymax - ymin) * scale, 2)
            cx = left + w / 2
            cy = top + h / 2
            markers.append(
                f'<div class="marker-bbox" style="left:{left}px;top:{top}px;width:{w}px;height:{h}px"></div>'
            )
            markers.append(f'<div class="marker-cross" style="left:{cx}px;top:{cy}px"></div>')
        elif isinstance(detail, (list, tuple)) and len(detail) == 2:
            cx, cy = detail
            cx *= scale
            cy *= scale
            markers.append(f'<div class="marker-cross" style="left:{cx}px;top:{cy}px"></div>')
    elif lower_type == "scroll" and detail:
        markers.append(f'<div class="marker-label">SCROLL {detail}</div>')
    elif lower_type == "press" and detail:
        markers.append(f'<div class="marker-label">PRESS {detail}</div>')
    elif lower_type == "type" and detail:
        markers.append(f'<div class="marker-label">TYPE</div>')
    elif lower_type == "stop":
        markers.append('<div class="marker-label">STOP</div>')

    label = action_type.upper()
    return {"width": display_width, "height": display_height, "markers": markers, "label": label}


def aggregate_metrics(results: List[dict]) -> Dict[str, object]:
    action_keys = ["CLICK", "LONG", "TYPE", "SCROLL", "PRESS", "STOP"]
    recorder = {
        "total": {"count": 0, "type_match": 0, "exact_match": 0, "hit": 0},
    }
    for key in action_keys:
        recorder[key] = {"count": 0, "type_match": 0, "exact_match": 0, "text_dist": []}

    for step in results:
        recorder["total"]["count"] += 1
        recorder["total"]["hit"] += 1 if step["format_hit"] else 0
        answer_type = step["answer"]["action_type"]
        if not answer_type:
            continue
        answer_key = answer_type.upper()
        if answer_key not in recorder:
            recorder[answer_key] = {"count": 0, "type_match": 0, "exact_match": 0, "text_dist": []}
        recorder[answer_key]["count"] += 1
        recorder[answer_key]["type_match"] += 1 if step["type_match"] else 0
        recorder[answer_key]["exact_match"] += 1 if step["exact_match"] else 0
        recorder["total"]["type_match"] += 1 if step["type_match"] else 0
        recorder["total"]["exact_match"] += 1 if step["exact_match"] else 0
        if step.get("text_dist") is not None:
            recorder[answer_key]["text_dist"].append(step["text_dist"])

    def _safe_rate(count: int, value: int) -> float:
        return round(value / count, 4) if count else 0.0

    summary: Dict[str, object] = {
        "total_steps": recorder["total"]["count"],
        "format_hit_rate": _safe_rate(recorder["total"]["count"], recorder["total"]["hit"]),
        "type_accuracy": _safe_rate(recorder["total"]["count"], recorder["total"]["type_match"]),
        "exact_accuracy": _safe_rate(recorder["total"]["count"], recorder["total"]["exact_match"]),
        "per_action": {},
    }
    for key in action_keys:
        per = recorder[key]
        summary["per_action"][key] = {
            "count": per["count"],
            "type_accuracy": _safe_rate(per["count"], per["type_match"]),
            "exact_accuracy": _safe_rate(per["count"], per["exact_match"]),
        }
        if per["text_dist"]:
            summary["per_action"][key]["avg_text_similarity"] = round(
                sum(per["text_dist"]) / len(per["text_dist"]), 4
            )
    return summary


def compute_episode_metrics(results: List[dict]) -> Dict[str, float]:
    episodes: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for step in results:
        episodes[(step["subset"], step["episode_id"])].append(step)
    if not episodes:
        return {"success_rate": 0.0, "goal_progress": 0.0}
    success_vals = []
    progress_vals = []
    for ep_steps in episodes.values():
        exact_matches = [1 if s["exact_match"] else 0 for s in ep_steps]
        success_vals.append(int(all(exact_matches)))
        progress_vals.append(sum(exact_matches) / len(ep_steps))
    return {
        "success_rate": round(sum(success_vals) / len(success_vals), 4),
        "goal_progress": round(sum(progress_vals) / len(progress_vals), 4),
    }


def generate_html(entries: List[dict], summary: Dict[str, object], html_path: Path, max_rows: int) -> None:
    if not entries:
        return
    html_path.parent.mkdir(parents=True, exist_ok=True)
    assets_dir = html_path.parent / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    def stash_asset(src: str, subdir: str) -> str:
        src_path = Path(src)
        if not src_path.exists():
            return src
        digest = hashlib.md5(str(src_path).encode()).hexdigest()[:12]
        target_dir = assets_dir / subdir
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / f"{digest}_{src_path.name}"
        if not target_path.exists():
            _safe_copy(src_path, target_path)
        return os.path.relpath(target_path, html_path.parent)

    rows = []
    for entry in entries[:max_rows]:
        step = entry["step"]
        result = entry["result"]
        rel_image = stash_asset(step["image_full_path"], "screens")
        status_cls = "ok" if result["exact_match"] else ("partial" if result["type_match"] else "fail")
        rel_json = stash_asset(entry["pred_path"], "json") if entry["pred_path"] else "#"
        gt_vis = entry["gt_visual"]
        pd_vis = entry["pred_visual"]
        gt_json = entry["gt_json"]
        # print(f"gt_json", gt_json)
        pd_json = entry["pred_json"]
        # print(f"pred_json", pd_json)
        rows.append(
            f"""
            <div class="entry {status_cls}">
              <div class="panels">
                <div class="panel">
                  <h4>Ground Truth</h4>
                  <div class="img-wrapper" style="width:{gt_vis['width']}px;height:{gt_vis['height']}px">
                    <img src=\"{rel_image}\" width=\"{gt_vis['width']}\" height=\"{gt_vis['height']}\" />
                    {''.join(gt_vis['markers'])}
                  </div>
                  <div class="action-label">{gt_vis['label']}</div>
                  <pre>{gt_json}</pre>
                </div>
                <div class="panel">
                  <h4>Prediction</h4>
                  <div class="img-wrapper" style="width:{pd_vis['width']}px;height:{pd_vis['height']}px">
                    <img src=\"{rel_image}\" width=\"{pd_vis['width']}\" height=\"{pd_vis['height']}\" />
                    {''.join(pd_vis['markers'])}
                  </div>
                  <div class="action-label">{pd_vis['label']}</div>
                  <pre>{pd_json}</pre>
                </div>
              </div>
              <div class="meta">
                <h3>{step['subset']} / {step['episode_id']} / step {step['step_id']}</h3>
                <p><strong>Instruction:</strong> {step['instruction']}</p>
                <p><strong>Exact Match:</strong> {result['exact_match']} | <strong>Type Match:</strong> {result['type_match']}</p>
                <p><a href="{rel_json}">Open prediction JSON</a></p>
              </div>
            </div>
            """
        )
    summary_block = json.dumps(summary, indent=2, ensure_ascii=False)
    html = f"""
    <html>
    <head>
      <meta charset="utf-8" />
      <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 0 16px 40px; }}
        header {{ padding: 16px 0; }}
        pre {{ background: #f5f5f5; padding: 12px; border-radius: 6px; overflow-x: auto; }}
        .nav-controls {{ display: flex; align-items: center; gap: 12px; margin: 16px 0; }}
        button {{ padding: 8px 16px; border-radius: 4px; border: 1px solid #ccc; cursor: pointer; }}
        button:disabled {{ opacity: 0.4; cursor: not-allowed; }}
        .entry {{ display: none; border: 1px solid #ccc; margin-bottom: 16px; padding: 12px; border-radius: 6px; flex-direction: column; gap: 12px; }}
        .entry.active {{ display: flex; }}
        .entry.ok {{ border-color: #2ca25f; }}
        .entry.fail {{ border-color: #de2d26; }}
        .entry.partial {{ border-color: #feb24c; }}
        .panels {{ display: flex; gap: 16px; flex-wrap: wrap; }}
        .panel {{ flex: 1; min-width: 320px; }}
        .img-wrapper {{ position: relative; background: #000; overflow: hidden; border-radius: 4px; border: 1px solid #ddd; }}
        .img-wrapper img {{ display: block; width: 100%; height: 100%; object-fit: contain; }}
        .marker-bbox {{ position: absolute; border: 2px solid #ff5252; pointer-events: none; }}
        .marker-cross {{ position: absolute; width: 16px; height: 16px; margin-left: -8px; margin-top: -8px; pointer-events: none; }}
        .marker-cross:before, .marker-cross:after {{ content: ""; position: absolute; background: #ff5252; }}
        .marker-cross:before {{ width: 16px; height: 2px; top: 7px; left: 0; }}
        .marker-cross:after {{ width: 2px; height: 16px; top: 0; left: 7px; }}
        .marker-label {{ position: absolute; top: 8px; left: 8px; background: rgba(0,0,0,0.6); color: #fff; padding: 4px 8px; border-radius: 4px; font-size: 12px; }}
        .action-label {{ margin-top: 8px; font-weight: bold; }}
        .meta p {{ margin: 4px 0; }}
        .counter {{ font-weight: bold; }}
      </style>
    </head>
    <body>
      <header>
        <h1>Prediction vs Ground Truth</h1>
        <pre>{summary_block}</pre>
        <div class="nav-controls">
          <button id="prev-btn">Prev</button>
          <span class="counter" id="entry-counter">0 / 0</span>
          <button id="next-btn">Next</button>
        </div>
      </header>
      <section id="entries">
        {''.join(rows)}
      </section>
      <script>
        const entries = Array.from(document.querySelectorAll('.entry'));
        const prevBtn = document.getElementById('prev-btn');
        const nextBtn = document.getElementById('next-btn');
        const counter = document.getElementById('entry-counter');
        let idx = 0;
        function update() {{
          entries.forEach((entry, i) => entry.classList.toggle('active', i === idx));
          counter.textContent = `${{idx + 1}} / ${{entries.length}}`;
          prevBtn.disabled = entries.length === 0;
          nextBtn.disabled = entries.length === 0;
        }}
        prevBtn.addEventListener('click', () => {{
          idx = (idx - 1 + entries.length) % entries.length;
          update();
        }});
        nextBtn.addEventListener('click', () => {{
          idx = (idx + 1) % entries.length;
          update();
        }});
        update();
      </script>
    </body>
    </html>
    """
    html_path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GUI agent predictions.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--pred-dir", type=Path, required=True, help="Directory containing predict JSON files.")
    parser.add_argument("--dataset-dir", help="Override DATA.DATA_DIR")
    parser.add_argument("--dataset-name", help="Override DATA.DATASET (e.g., amex)")
    parser.add_argument("--split", help="Override DATA.SPLIT")
    parser.add_argument("--demo-mode", default=None, help="Demo mode used during prediction (default from config).")
    parser.add_argument(
        "--screen-mode",
        choices=["tag"],
        help="Screen representation used during predict (screen-tag images only).",
    )
    parser.add_argument("--model-name", help="Model name for evaluator traits (default from config).")
    parser.add_argument("--click-mode", choices=["bbox", "point"], help="Override click evaluation mode.")
    parser.add_argument("--ratio", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--html-report", type=Path, help="Optional HTML report path.")
    parser.add_argument("--open-browser", action="store_true", help="Open the HTML report in the default browser when done.")
    parser.add_argument(
        "--serve", action="store_true", help="Start a lightweight HTTP server to host the report and print a localhost URL."
    )
    parser.add_argument("--serve-port", type=int, default=8992, help="Port used when --serve is enabled.")
    parser.add_argument("--serve-host", default="0.0.0.0", help="Host/IP to bind when serving (default 0.0.0.0).")
    parser.add_argument("--max-html", type=int, default=200)
    parser.add_argument("--metrics-json", type=Path, help="Optional JSON file to store metrics.")
    parser.add_argument("--verbose", action="store_true", help="Verbose evaluation logs.")
    args = parser.parse_args()

    apply_dataset_alias(args)
    cfg = load_config(args.config, overrides=build_overrides(args))
    demo_mode = (cfg.DEMO_MODE if not args.demo_mode else args.demo_mode).upper()
    if not cfg.DATA.USE_SCREEN_TAG:
        raise ValueError("Text-only screen evaluations are no longer supported; please enable SCREEN_TAG.")
    screen_mode = args.screen_mode or "tag"
    model_name = args.model_name or cfg.MODEL.NAME

    dataset_obj = build_dataset(cfg, ratio=args.ratio, double_sample=False)
    dataset = list(dataset_obj)
    if args.limit:
        dataset = dataset[: args.limit]

    evaluator = ActionEvaluator(
        demo_mode,
        screen_mode,
        model=model_name,
        click_mode_override=args.click_mode,
        verbose=args.verbose,
    )

    pred_dir = args.pred_dir.resolve()
    if not pred_dir.exists():
        raise FileNotFoundError(f"Prediction directory {pred_dir} does not exist.")

    step_results_all: List[dict] = []
    step_results_completed: List[dict] = []
    html_entries: List[dict] = []
    completed_count = 0
    for step in dataset:
        pred_path = prediction_path(step, pred_dir)
        pred_payload = {}
        if pred_path.exists():
            try:
                with open(pred_path, "r", encoding="utf-8") as handle:
                    pred_payload = json.load(handle)
            except json.JSONDecodeError:
                pred_payload = {}
        result = evaluator(step, pred_payload)
        step_results_all.append(result)
        if has_prediction(pred_payload, demo_mode, screen_mode):
            completed_count += 1
            step_results_completed.append(result)
            mode_entry = pred_payload["action_predict"][demo_mode][screen_mode]
            gt_vis = build_visual_overlay(step["image_full_path"], result["answer"]["action_type"], result["answer"]["action_detail"])
            pd_vis = build_visual_overlay(step["image_full_path"], result["pred"]["action_type"], result["pred"]["action_detail"])
            html_entries.append(
                {
                    "step": step,
                    "result": result,
                    "pred_entry": mode_entry,
                    "pred_path": pred_path,
                    "relevant_ui_path": pred_payload.get("relevant_ui_path"),
                    "gt_visual": gt_vis,
                    "pred_visual": pd_vis,
                    "gt_json": _format_action_json(result["answer"]["action_type"], result["answer"]["action_detail"]),
                    "pred_json": _format_action_json(result["pred"]["action_type"], result["pred"]["action_detail"]),
                }
            )

    overall_metrics = aggregate_metrics(step_results_all)
    overall_metrics["episodes"] = compute_episode_metrics(step_results_all)
    completed_metrics = aggregate_metrics(step_results_completed) if step_results_completed else {}
    if step_results_completed:
        completed_metrics["episodes"] = compute_episode_metrics(step_results_completed)

    print("=== Overall (all steps) ===")
    print(json.dumps(overall_metrics, indent=2, ensure_ascii=False))
    if step_results_completed:
        print("\n=== Completed Predictions ===")
        print(json.dumps(completed_metrics, indent=2, ensure_ascii=False))
    else:
        print("\n[warn] No completed predictions found.")

    if args.metrics_json:
        args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.metrics_json, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "overall": overall_metrics,
                    "completed": completed_metrics,
                    "total_steps": len(step_results_all),
                    "completed_steps": completed_count,
                },
                handle,
                indent=2,
                ensure_ascii=False,
            )

    server_url = None
    if args.html_report:
        generate_html(
            html_entries,
            {
                "overall": overall_metrics,
                "completed": completed_metrics,
                "total_steps": len(step_results_all),
                "completed_steps": completed_count,
            },
            args.html_report,
            args.max_html,
        )
        resolved_report = args.html_report.resolve()
        print(f"\nInteractive report ready: {resolved_report.as_uri()}")
        if args.serve:
            import http.server
            import socketserver
            import threading
            import time
            import webbrowser

            serve_dir = resolved_report.parent
            handler = http.server.SimpleHTTPRequestHandler
            os.chdir(serve_dir)
            try:
                httpd = socketserver.TCPServer((args.serve_host, args.serve_port), handler)
            except OSError as exc:
                print(f"[error] Failed to start server on {args.serve_host}:{args.serve_port} ({exc})")
                return
            link_host = args.serve_host if args.serve_host not in ("0.0.0.0", "::") else "127.0.0.1"
            server_url = f"http://{link_host}:{args.serve_port}/{resolved_report.name}"

            def _serve():
                with httpd:
                    httpd.serve_forever()

            thread = threading.Thread(target=_serve, daemon=True)
            thread.start()
            print(f"Web server running at: {server_url}")
            if args.open_browser:
                try:
                    webbrowser.open(server_url)
                except Exception as exc:
                    print(f"[warn] Failed to open browser: {exc}")
            print("Press Ctrl+C to stop serving...")
            try:
                while True:
                    time.sleep(0.5)
            except KeyboardInterrupt:
                print("\nStopping server...")
                httpd.shutdown()
                thread.join()
            return

        if args.open_browser:
            import webbrowser

            try:
                webbrowser.open(resolved_report.as_uri())
            except Exception as exc:
                print(f"[warn] Failed to open browser: {exc}")


if __name__ == "__main__":
    main()
