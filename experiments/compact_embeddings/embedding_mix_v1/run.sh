#!/usr/bin/env bash
set -euo pipefail
cd /home/Yue_Ziran/workspace/ana-embedding-mix-v1
embedding_model=${1:?model required}
export CUDA_VISIBLE_DEVICES=${2:?GPU index required}
case "$embedding_model" in embedding_mix_learned) ;; *) exit 2 ;; esac
case "$(hostname -s)" in
    exp15) embedding_python=/home/Yue_Ziran/.venv/bin/python ;;
    *) exit 2 ;;
esac
export ANA_EXTRA_PYTHONPATH=/home/Yue_Ziran/workspace/ana-benes/python-deps
export PYTHONPATH="$PWD/src${ANA_EXTRA_PYTHONPATH:+:$ANA_EXTRA_PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TMPDIR="$PWD/tmp"
export ANA_DATA=/mango/homes/YUE_Ziran/workspace/ana-embedding-mix-v1/data
export XDG_CACHE_HOME=/mango/homes/YUE_Ziran/workspace/ana-embedding-mix-v1/cache/xdg
export HF_HOME=/mango/homes/YUE_Ziran/workspace/ana-embedding-mix-v1/cache/huggingface
export TORCH_HOME=/mango/homes/YUE_Ziran/workspace/ana-embedding-mix-v1/cache/torch
export ANA_MAX_GPU_MIB=${ANA_MAX_GPU_MIB:-9216}
mkdir -p logs reports "$TMPDIR" "$XDG_CACHE_HOME" "$HF_HOME" "$TORCH_HOME"
exec 9>"logs/gpu${CUDA_VISIBLE_DEVICES}.lock"
flock -n 9 || { echo 'This task already owns this GPU.'; exit 1; }
trap 'echo "EMBEDDING_MIX_FAILED at line $LINENO"' ERR
printf 'Owner PID: %s; model: %s; GPU: %s\n' "$$" "$embedding_model" "$CUDA_VISIBLE_DEVICES"
date --iso-8601=seconds
"$embedding_python" -B run_cell.py "$embedding_model"
"$embedding_python" -B report.py multi30k
date --iso-8601=seconds
echo EMBEDDING_MIX_WORKER_COMPLETE
