#!/bin/bash
cd /media/caizf/pengru/deps_san
#python sdsa_matrix.py  --dp-files ../data/iwslt14_de_en/test.dp.de ../data/iwslt14_de_en/train.dp.de \
#../data/iwslt14_de_en/valid.dp.de

python preprocess_matrix.py --source-lang de --target-lang en --trainpref ../data/iwslt14_de_en/train \
--validpref ../data/iwslt14_de_en/valid --testpref ../data/iwslt14_de_en/test \
--destdir ../data/iwslt14_de_en/sdsa --matrix-suffix sdsa --joined-dictionary \
--thresholdtgt 0 --thresholdsrc 0 --nwordstgt 60000 --nwordssrc 60000 --workers 16