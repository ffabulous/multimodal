#!/usr/bin/env fish

conda activate video

python train.py \
    --cache-dataset \
    --output-dir dump/0 \
    --log-dir logs/0 \
    --gpu 0