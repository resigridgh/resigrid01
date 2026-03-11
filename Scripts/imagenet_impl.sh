nohup python imagenet_impl.py \
--epochs 10000 \
--train_ratio 0.1 \
--val_ratio 0.05 \
> training.log 2>&1 &
