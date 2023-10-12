#!/bin/bash
cd /media/caizf/pengru/deps_san
CUDA_VISIBLE_DEVICES=0  python train.py ../data/iwslt14_de_en/sdsa \
--task translation_sdsa --matrix-suffix sdsa --arch transformer_sdsa_wmt_en_de --share-all-embeddings --dropout 0.1 \
--enc-sdsa-heads 8 8 8 0 0 0 --prob-q 0.0 --win-k 0.0 --weight-fn normal --weight-sigma 1 --weight-usage mult \
--optimizer adam --adam-betas 0.9,0.98 --clip-norm 0.0 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 \
--warmup-updates 8000 --lr 0.0007 --min-lr 1e-09 --ddp-backend=no_c10d --criterion label_smoothed_cross_entropy \
--label-smoothing 0.1 --weight-decay 0.0 --max-tokens 4096 --save-dir ../checkpoints/iwslt14_de_en/sdsa --update-freq 1 --no-progress-bar \
--log-format json --log-interval 1000 --save-interval-updates 1000 --keep-interval-updates 1 --max-update 60000
