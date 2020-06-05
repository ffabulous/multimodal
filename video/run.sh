#!/usr/bin/env fish

conda activate mm

python train.py \
	--cache-dataset \
	--train-dir train \
	--valid-dir valid \
	--output-dir dump/0 \
	--log-dir logs/0 \
	--log-freq 50 \
	--epochs 100 \
	--pretrained \
	--fine-tune \
	--batch-size 32 \
	--gpu 0
