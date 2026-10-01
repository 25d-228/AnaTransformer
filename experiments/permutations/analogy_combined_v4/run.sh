#!/usr/bin/env bash
set -euo pipefail
export ANA_SERVER=${ANA_SERVER:-/home/Yue_Ziran/workspace/ana-analogy-combined-v4}
export ANA_NAS=${ANA_NAS:-/mango/homes/YUE_Ziran/workspace/ana-analogy-combined-v4}
cd "$ANA_SERVER"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${1:?GPU index required}
queue_file=${2:?queue file required}
queue_run_cell=${ANA_RUN_CELL:-run_cell.py}
queue_report=${ANA_REPORT:-report.py}
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
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TMPDIR="$PWD/tmp"
export ANA_DATA="$ANA_NAS/data"
export XDG_CACHE_HOME="$ANA_NAS/cache/xdg"
export HF_HOME="$ANA_NAS/cache/huggingface"
export TORCH_HOME="$ANA_NAS/cache/torch"
export CUDA_CACHE_PATH="$ANA_NAS/cache/cuda"
queue_default_cap=${ANA_MAX_GPU_MIB:-9216}
queue_default_checkpointing=${ANA_ACTIVATION_CHECKPOINTING:-0}
mkdir -p logs reports "$TMPDIR" "$XDG_CACHE_HOME" "$HF_HOME" "$TORCH_HOME" "$CUDA_CACHE_PATH"
exec 9>"logs/gpu${CUDA_VISIBLE_DEVICES}.lock"
if [[ ${ANA_WAIT_GPU_LOCK:-0} == 1 ]]; then
    echo "Waiting for the previous GPU queue, if still running."
    flock 9
else
    flock -n 9 || { echo 'This task already owns this GPU.'; exit 1; }
fi
queue_failures=0
printf 'Owner PID: %s; host: %s; GPU: %s; queue: %s\n' \
    "$$" "$queue_host" "$CUDA_VISIBLE_DEVICES" "$queue_file"
while read -r queue_corpus queue_model; do
    [[ -z $queue_corpus || $queue_corpus == \#* ]] && continue
    case "$queue_corpus" in
        multi30k|multi30k_enfr)
            export ANA_MICRO_BATCH_SIZE=${ANA_TRANSLATION_MICRO_BATCH_SIZE:-128}
            export ANA_MAX_GPU_MIB=${ANA_TRANSLATION_MAX_GPU_MIB:-$queue_default_cap}
            export ANA_ACTIVATION_CHECKPOINTING=${ANA_TRANSLATION_ACTIVATION_CHECKPOINTING:-$queue_default_checkpointing}
            export ANA_DECODE_BATCH_SIZE=${ANA_TRANSLATION_DECODE_BATCH_SIZE:-32}
            ;;
        cogs)
            export ANA_MICRO_BATCH_SIZE=${ANA_COGS_MICRO_BATCH_SIZE:-64}
            export ANA_MAX_GPU_MIB=${ANA_COGS_MAX_GPU_MIB:-$queue_default_cap}
            export ANA_ACTIVATION_CHECKPOINTING=${ANA_COGS_ACTIVATION_CHECKPOINTING:-$queue_default_checkpointing}
            export ANA_DECODE_BATCH_SIZE=${ANA_COGS_DECODE_BATCH_SIZE:-32}
            ;;
        iwslt14)
            export ANA_MICRO_BATCH_SIZE=${ANA_IWSLT_MICRO_BATCH_SIZE:-20}
            export ANA_MAX_GPU_MIB=${ANA_IWSLT_MAX_GPU_MIB:-$queue_default_cap}
            export ANA_ACTIVATION_CHECKPOINTING=${ANA_IWSLT_ACTIVATION_CHECKPOINTING:-0}
            export ANA_DECODE_BATCH_SIZE=${ANA_IWSLT_DECODE_BATCH_SIZE:-32}
            ;;
        *) echo "Unknown corpus: $queue_corpus"; exit 2 ;;
    esac
    case "$queue_model" in
        combo|combo_wide|compact_q|compact_qkv|combo_crosskv|combo_selfqk|balanced_qkv|gated_qkv|pre_crossq|pre_lowrank|balanced_gated|shared_bottleneck|diagonal_shortcuts|compact_qkv_no_analogy|balanced_qkv_no_analogy|d_router_03|d_router_10|d_cross_focus|compact_qkv_clean) ;;
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
    printf 'START corpus=%s model=%s at=%s\n' \
        "$queue_corpus" "$queue_model" "$(date --iso-8601=seconds)"
    queue_code=0
    for queue_attempt in 1 2; do
        if "$queue_python" -B "$queue_run_cell" "$queue_corpus" "$queue_model" >> "$queue_log" 2>&1; then
            queue_code=0
            break
        else
            queue_code=$?
        fi
        if (( queue_attempt == 1 && (queue_code == 139 || queue_code == 134) )); then
            printf 'RETRY_FROM_CHECKPOINT corpus=%s model=%s exit=%s at=%s\n' \
                "$queue_corpus" "$queue_model" "$queue_code" "$(date --iso-8601=seconds)"
        else
            break
        fi
    done
    if (( queue_code == 0 )); then
        printf 'COMPLETED corpus=%s model=%s at=%s\n' \
            "$queue_corpus" "$queue_model" "$(date --iso-8601=seconds)"
        if ! "$queue_python" -B "$queue_report" "$queue_corpus" >> "$queue_log" 2>&1; then
            echo "REPORT_FAILED corpus=$queue_corpus model=$queue_model"
            queue_failures=$((queue_failures + 1))
        fi
    else
        printf 'CELL_FAILED corpus=%s model=%s exit=%s at=%s\n' \
            "$queue_corpus" "$queue_model" "$queue_code" "$(date --iso-8601=seconds)"
        queue_failures=$((queue_failures + 1))
    fi
done < "$queue_file"
if ! "$queue_python" -B "$queue_report" all >> "logs/final_report_gpu${CUDA_VISIBLE_DEVICES}.log" 2>&1; then
    echo 'FINAL_REPORT_FAILED'
    queue_failures=$((queue_failures + 1))
fi
printf 'ANALOGY_COMBINED_QUEUE_FINISHED failures=%s at=%s\n' \
    "$queue_failures" "$(date --iso-8601=seconds)"
(( queue_failures == 0 ))
