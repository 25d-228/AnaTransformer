#!/usr/bin/env bash
set -euo pipefail
cd /home/Yue_Ziran/workspace/ana-power-consistency-iwslt-v1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${1:?GPU index required}
consistency_model=${2:-embedding_power_consistency}
if (( $# > 2 )); then
    echo 'Usage: run.sh GPU [embedding_power_consistency|embedding_linear]'
    exit 2
fi
case "$consistency_model" in embedding_power_consistency|embedding_linear) ;; *) exit 2 ;; esac
: "${ANA_MAX_GPU_MIB:?Set ANA_MAX_GPU_MIB explicitly after checking GPU capacity}"
case "$(hostname -s)" in
    exp18) consistency_python=/home/Yue_Ziran/.venv/bin/python ;;
    *) exit 2 ;;
esac
export PYTHONPATH="$PWD/src"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TMPDIR="$PWD/tmp"
export ANA_DATA=/mango/homes/YUE_Ziran/workspace/ana-power-consistency-iwslt-v1/data
export XDG_CACHE_HOME=/mango/homes/YUE_Ziran/workspace/ana-power-consistency-iwslt-v1/cache/xdg
export HF_HOME=/mango/homes/YUE_Ziran/workspace/ana-power-consistency-iwslt-v1/cache/huggingface
export TORCH_HOME=/mango/homes/YUE_Ziran/workspace/ana-power-consistency-iwslt-v1/cache/torch
export ANA_MAX_GPU_MIB
mkdir -p logs reports "$TMPDIR" "$XDG_CACHE_HOME" "$HF_HOME" "$TORCH_HOME"
exec 9>"logs/gpu${CUDA_VISIBLE_DEVICES}.lock"
flock -n 9 || { echo 'This task already owns this GPU.'; exit 1; }
trap 'echo "POWER_CONSISTENCY_IWSLT_FAILED at line $LINENO"' ERR
printf 'Owner PID: %s; model: %s; GPU: %s\n' "$$" "$consistency_model" "$CUDA_VISIBLE_DEVICES"
date --iso-8601=seconds
"$consistency_python" -B run_cell.py "$consistency_model"
"$consistency_python" -B report.py iwslt14
date --iso-8601=seconds
echo POWER_CONSISTENCY_IWSLT_WORKER_COMPLETE
