#!/usr/bin/env bash
set -euo pipefail
export ANA_SERVER=${ANA_SERVER:-/home/Yue_Ziran/workspace/ana-adaptive-power-balanced-v1}
export ANA_NAS=${ANA_NAS:-/mango/homes/YUE_Ziran/workspace/ana-adaptive-power-balanced-v1}
cd "$ANA_SERVER"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${1:?GPU index required}
queue_corpus=${2:?corpus required}
queue_model=${3:?model required}
if (( $# != 3 )); then
    echo 'Usage: bash run.sh GPU CORPUS MODEL'
    exit 2
fi
case "$queue_corpus" in multi30k|multi30k_enfr|cogs) ;; *) exit 2 ;; esac
case "$queue_model" in embedding_adaptive_power_balanced) ;; *) exit 2 ;; esac
if [[ ! $CUDA_VISIBLE_DEVICES =~ ^[0-9]+$ ]]; then
    echo 'One numeric GPU index is required.'
    exit 2
fi
queue_host=$(hostname -s)
if [[ -n ${ANA_PYTHON:-} ]]; then
    queue_python=$ANA_PYTHON
else
    case "$queue_host" in
        exp16|exp17) queue_python=/home/Yue_Ziran/workspace/ana-runtime/.venv/bin/python ;;
        exp14|exp15|exp18) queue_python=/home/Yue_Ziran/.venv/bin/python ;;
        *) echo 'Set ANA_PYTHON for this host.'; exit 2 ;;
    esac
fi
if [[ $queue_host == exp15 && -z ${ANA_EXTRA_PYTHONPATH+x} ]]; then
    export ANA_EXTRA_PYTHONPATH=/home/Yue_Ziran/workspace/ana-benes/python-deps
fi
export PYTHONPATH="$PWD/src${ANA_EXTRA_PYTHONPATH:+:$ANA_EXTRA_PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}
export TMPDIR=${TMPDIR:-$PWD/tmp}
export ANA_DATA=${ANA_DATA:-$ANA_NAS/data}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$ANA_NAS/cache/xdg}
export HF_HOME=${HF_HOME:-$ANA_NAS/cache/huggingface}
export TORCH_HOME=${TORCH_HOME:-$ANA_NAS/cache/torch}
export CUDA_CACHE_PATH=${CUDA_CACHE_PATH:-$ANA_NAS/cache/cuda}
export ANA_MAX_GPU_MIB=${ANA_MAX_GPU_MIB:-8192}
mkdir -p logs reports "$TMPDIR" "$XDG_CACHE_HOME" "$HF_HOME" "$TORCH_HOME" "$CUDA_CACHE_PATH"
exec 9>"logs/gpu${CUDA_VISIBLE_DEVICES}.lock"
flock -n 9 || { echo 'This task already owns this GPU.'; exit 1; }
trap 'echo "BALANCED_ADAPTIVE_POWER_FAILED model=$queue_model at line $LINENO"' ERR
queue_log="logs/${queue_corpus}_${queue_model}.log"
printf 'Owner PID: %s; corpus: %s; GPU: %s; model: %s\n' \
    "$$" "$queue_corpus" "$CUDA_VISIBLE_DEVICES" "$queue_model"
printf 'Starting %s/%s; log: %s\n' "$queue_corpus" "$queue_model" "$queue_log"
date --iso-8601=seconds
"$queue_python" -B run_cell.py "$queue_corpus" "$queue_model" >> "$queue_log" 2>&1
if [[ -f report.py ]]; then
    "$queue_python" -B report.py "$queue_corpus" >> "$queue_log" 2>&1
fi
printf 'Completed %s/%s\n' "$queue_corpus" "$queue_model"
date --iso-8601=seconds
echo BALANCED_ADAPTIVE_POWER_WORKER_COMPLETE
