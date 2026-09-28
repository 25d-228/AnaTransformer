#!/usr/bin/env bash
set -euo pipefail
export ANA_SERVER=${ANA_SERVER:-/home/Yue_Ziran/workspace/ana-context-power-v1}
export ANA_NAS=${ANA_NAS:-/mango/homes/YUE_Ziran/workspace/ana-context-power-v1}
cd "$ANA_SERVER"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${1:?GPU index required}
queue_file=${2:?queue file required}
if (( $# != 2 )) || [[ ! $CUDA_VISIBLE_DEVICES =~ ^[0-9]+$ ]]; then
    echo 'Usage: bash run.sh GPU QUEUE_FILE'
    exit 2
fi
queue_host=$(hostname -s)
case "$queue_host" in
    exp16|exp17) queue_python=/home/Yue_Ziran/workspace/ana-runtime/.venv/bin/python ;;
    exp14|exp15|exp18) queue_python=/home/Yue_Ziran/.venv/bin/python ;;
    *) echo 'Unsupported host'; exit 2 ;;
esac
if [[ $queue_host == exp15 ]]; then
    export ANA_EXTRA_PYTHONPATH=${ANA_EXTRA_PYTHONPATH:-/home/Yue_Ziran/workspace/ana-benes/python-deps}
fi
export PYTHONPATH="$PWD/src${ANA_EXTRA_PYTHONPATH:+:$ANA_EXTRA_PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TMPDIR="$PWD/tmp"
export ANA_DATA="$ANA_NAS/data"
export XDG_CACHE_HOME="$ANA_NAS/cache/xdg"
export HF_HOME="$ANA_NAS/cache/huggingface"
export TORCH_HOME="$ANA_NAS/cache/torch"
export CUDA_CACHE_PATH="$ANA_NAS/cache/cuda"
export ANA_MAX_GPU_MIB=${ANA_MAX_GPU_MIB:-9216}
mkdir -p logs reports "$TMPDIR" "$XDG_CACHE_HOME" "$HF_HOME" "$TORCH_HOME" "$CUDA_CACHE_PATH"
exec 9>"logs/gpu${CUDA_VISIBLE_DEVICES}.lock"
flock -n 9 || { echo 'This task already owns this GPU.'; exit 1; }
queue_failures=0
printf 'Owner PID: %s; host: %s; GPU: %s; queue: %s\n' \
    "$$" "$queue_host" "$CUDA_VISIBLE_DEVICES" "$queue_file"
while read -r queue_corpus queue_model; do
    [[ -z $queue_corpus || $queue_corpus == \#* ]] && continue
    case "$queue_corpus" in
        multi30k|multi30k_enfr) export ANA_MICRO_BATCH_SIZE=128 ;;
        cogs) export ANA_MICRO_BATCH_SIZE=64 ;;
        *) echo "Unknown corpus: $queue_corpus"; exit 2 ;;
    esac
    case "$queue_model" in
        context_power_prediction|context_power_prediction_analogy|context_power_features|context_power_features_analogy|context_power_features_fixed) ;;
        *) echo "Unknown model: $queue_model"; exit 2 ;;
    esac
    while true; do
        queue_free=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.free --format=csv,noheader,nounits)
        queue_ram=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
        if (( queue_free >= ANA_MAX_GPU_MIB + 1024 && queue_ram >= 8 * 1024 * 1024 )); then
            break
        fi
        printf 'WAITING_FOR_RESOURCES corpus=%s model=%s free_mib=%s\n' \
            "$queue_corpus" "$queue_model" "$queue_free"
        sleep 60
    done
    queue_log="logs/${queue_corpus}_${queue_model}.log"
    printf 'START corpus=%s model=%s at=%s\n' "$queue_corpus" "$queue_model" "$(date --iso-8601=seconds)"
    if "$queue_python" -B run_cell.py "$queue_corpus" "$queue_model" >> "$queue_log" 2>&1; then
        printf 'COMPLETED corpus=%s model=%s at=%s\n' "$queue_corpus" "$queue_model" "$(date --iso-8601=seconds)"
        if ! "$queue_python" -B report.py "$queue_corpus" >> "$queue_log" 2>&1; then
            echo "REPORT_FAILED corpus=$queue_corpus model=$queue_model"
            queue_failures=$((queue_failures + 1))
        fi
    else
        queue_code=$?
        printf 'CELL_FAILED corpus=%s model=%s exit=%s at=%s\n' \
            "$queue_corpus" "$queue_model" "$queue_code" "$(date --iso-8601=seconds)"
        queue_failures=$((queue_failures + 1))
    fi
done < "$queue_file"
if ! "$queue_python" -B report.py all >> "logs/final_report_gpu${CUDA_VISIBLE_DEVICES}.log" 2>&1; then
    echo 'FINAL_REPORT_FAILED'
    queue_failures=$((queue_failures + 1))
fi
printf 'CONTEXT_POWER_QUEUE_FINISHED failures=%s at=%s\n' "$queue_failures" "$(date --iso-8601=seconds)"
(( queue_failures == 0 ))
