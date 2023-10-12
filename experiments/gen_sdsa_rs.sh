#!/bin/bash
cd /media/caizf/pengru/deps_san
CUDA_VISIBLE_DEVICES=0 python generate.py ../data/iwslt14_de_en/sdsa \
--task translation_sdsa --matrix-suffix sdsa --path ../checkpoints/iwslt14_de_en/sdsa_rs/checkpoint_best.pt \
--gen-subset test --beam 5 --remove-bpe --batch-size 128 --lenpen 0.6 --results-path ../checkpoints/iwslt14_de_en/sdsa_rs

#CUDA_VISIBLE_DEVICES=0 python generate.py ../data/iwslt14_de_en/sdsa \
#--task translation_sdsa --matrix-suffix sdsa --path ../checkpoints/iwslt14_de_en/sdsa_rs/checkpoint_best.pt \
#--gen-subset valid --beam 5 --remove-bpe --batch-size 128 --lenpen 0.6 --results-path ../checkpoints/iwslt14_de_en/sdsa_rs