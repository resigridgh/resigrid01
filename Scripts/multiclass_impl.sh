#!/bin/bash
set -e

KEYWORD="hw02"

for i in 1 2 3 4 5
do
  python scripts/multiclass_impl.py \
    --data_path data/Android_Malware.csv \
    --eta 0.001 \
    --epochs 50 \
    --batch_size 2048 \
    --optimizer adam \
    --test_size 0.2 \
    --keyword ${KEYWORD}
done

python scripts/multiclass_eval.py --keyword ${KEYWORD}
