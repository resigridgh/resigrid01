#!/usr/bin/env bash
set -e

export PYTHONPATH=src

ZIP_PATH="/data/CPE_487-587/img_align_celeba.zip"
OUT_DIR="outputs/genmodels"
EVAL_DIR="outputs/gen_eval"
LOG_DIR="logs"
mkdir -p "${LOG_DIR}"

nohup python Scripts/genmodel_impl.py \
  --zip-path "${ZIP_PATH}" \
  --model all \
  --epochs 100 \
  --batch-size 128 \
  --lr 1e-4 \
  --latent-dim 128 \
  --train-ratio 0.05 \
  --onnx-every 5 \
  --diffusion-steps 200 \
  --device cuda \
  --out-dir "${OUT_DIR}" \
  > "${LOG_DIR}/genmodel_train.log" 2>&1 &

echo "Training started in background."
echo "Log file: ${LOG_DIR}/genmodel_train.log"
echo "Use: tail -f ${LOG_DIR}/genmodel_train.log"
