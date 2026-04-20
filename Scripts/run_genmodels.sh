#!/bin/bash

cd ~/resigrid01 || exit 1
export PYTHONPATH=src

TS=$(date +"%d-%m-%Y_%I-%M%p")

MODEL=${1:-all}
EPOCHS=${2:-1000}
GPU=${3:-0}

echo "====================================="
echo "Starting run at $TS"
echo "Model: $MODEL"
echo "Epochs: $EPOCHS"
echo "GPU: $GPU"
echo "====================================="

# -------------------------
# TRAINING
# -------------------------
CUDA_VISIBLE_DEVICES=$GPU python -u Scripts/genmodel_impl.py \
    --model $MODEL \
    --epochs $EPOCHS \
    --train-ratio 0.9 \
    --data-path ~/datasets/img_align_celeba/img_align_celeba \
    --batch-size 128 \
    --num-workers 8 \
    --device cuda \
    --onnx-every 100 \
    > train_${MODEL}_${TS}.log 2>&1

# -------------------------
# INFERENCE
# -------------------------
CUDA_VISIBLE_DEVICES=$GPU python -u Scripts/genmodel_inference.py \
    --model-dir outputs/genmodels \
    --latent-dim 128 \
    --diffusion-steps 200 \
    --num-samples 25 \
    --device cuda \
    --out-dir outputs/gen_eval_${TS} \
    > infer_${MODEL}_${TS}.log 2>&1

echo "Completed run at $(date)"
