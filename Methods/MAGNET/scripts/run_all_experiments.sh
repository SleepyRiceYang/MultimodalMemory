#!/usr/bin/env bash
# Exhaustive launcher that enumerates every demo-mode / memory / click-mode combination.
# Customize via environment variables without editing the script:
#   DRY_RUN=1 ./scripts/run_all_experiments.sh  # print commands only
#   OUTPUT_ROOT=artifacts/runs/ablation ./scripts/run_all_experiments.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

DRY_RUN="${DRY_RUN:-0}"
NUM_WORKERS="${NUM_WORKERS:-64}"
RATIO="${RATIO:-1.0}"
LIMIT="${LIMIT:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/artifacts/runs/exhaustive}"
AUTO_EVAL="${AUTO_EVAL:-1}"

declare -A DATASET_SPLITS=(
    [aitz]="test"
    [odyssey]="test_sample val_sample"
    [amex]="val_id_app val_ood_app val_ood_region"
)

DEFAULT_CONFIG="${DEFAULT_CONFIG:-configs/presets/coat/config_gemini_android.yaml}"
declare -A DATASET_CONFIGS=(
    [aitz]="${BASE_CONFIG_AITZ:-$DEFAULT_CONFIG}"
    [odyssey]="${BASE_CONFIG_ODYSSEY:-$DEFAULT_CONFIG}"
    [amex]="${BASE_CONFIG_AMEX:-$DEFAULT_CONFIG}"
)

FLOW_MODELS=(qwen2.5-vl-32b atlas-base-7b atlas-pro-7b gui-r1-7b ui-venus-navi-7b gemini-2.5-pro gpt-4o)
PREDICT_MODELS=("${FLOW_MODELS[@]}")
CLICK_MODES=(bbox point)
FLOW_PROCEDURAL_FLAGS=("on" "off")
PREDICT_STATIONARY_FLAGS=("on" "off")

declare -A E2E_MODEL_GROUPS=(
    [coa]="gemini-2.5-pro qwen2.5-vl-32b"
    [cot]="gemini-2.5-pro qwen2.5-vl-32b"
    [atlas]="atlas-base-7b atlas-pro-7b"
    [guir1]="gui-r1-7b"
    [ui_venus]="ui-venus-navi-7b"
)

run_cmd() {
    echo "+ $*"
    if [[ "${DRY_RUN}" == "0" ]]; then
        "$@"
    fi
}

launch_coat() {
    local dataset="$1"
    local split="$2"
    local base_config="$3"
    for flow_model in "${FLOW_MODELS[@]}"; do
        for predict_model in "${PREDICT_MODELS[@]}"; do
            for flow_proc in "${FLOW_PROCEDURAL_FLAGS[@]}"; do
                for predict_stationary in "${PREDICT_STATIONARY_FLAGS[@]}"; do
                    for click in "${CLICK_MODES[@]}"; do
                        local cmd=(python scripts/workflows/run_all_model_pairs.py
                            --base-config "${base_config}"
                            --output-root "${OUTPUT_ROOT}"
                            --dataset "${dataset}"
                            --subsets "${split}"
                            --ratio "${RATIO}"
                            --limit "${LIMIT}"
                            --num-workers "${NUM_WORKERS}"
                            --demo-mode coat
                            --predict-click-mode "${click}"
                            --models "${flow_model}" "${predict_model}"
                        )
                        if [[ "${flow_proc}" == "on" ]]; then
                            cmd+=(--flow-procedural)
                        else
                            cmd+=(--no-flow-procedural)
                        fi
                        if [[ "${predict_stationary}" == "on" ]]; then
                            cmd+=(--predict-stationary)
                        else
                            cmd+=(--no-predict-stationary)
                        fi
                        if [[ "${AUTO_EVAL}" == "1" ]]; then
                            cmd+=(--auto-eval)
                        fi
                        run_cmd "${cmd[@]}"
                    done
                done
            done
        done
    done
}

launch_e2e() {
    local dataset="$1"
    local split="$2"
    local base_config="$3"
    for demo_mode in "${!E2E_MODEL_GROUPS[@]}"; do
        for model in ${E2E_MODEL_GROUPS[${demo_mode}]}; do
            for click in "${CLICK_MODES[@]}"; do
                local cmd=(python scripts/workflows/run_all_model_pairs.py
                    --base-config "${base_config}"
                    --output-root "${OUTPUT_ROOT}"
                    --dataset "${dataset}"
                    --subsets "${split}"
                    --ratio "${RATIO}"
                    --limit "${LIMIT}"
                    --num-workers "${NUM_WORKERS}"
                    --demo-mode "${demo_mode}"
                    --predict-click-mode "${click}"
                    --models "${model}"
                )
                if [[ "${AUTO_EVAL}" == "1" ]]; then
                    cmd+=(--auto-eval)
                fi
                run_cmd "${cmd[@]}"
            done
        done
    done
}

for dataset in "${!DATASET_SPLITS[@]}"; do
    for split in ${DATASET_SPLITS[${dataset}]}; do
        base_cfg="${DATASET_CONFIGS[${dataset}]}"
        launch_coat "${dataset}" "${split}" "${base_cfg}"
        launch_e2e "${dataset}" "${split}" "${base_cfg}"
    done
done
