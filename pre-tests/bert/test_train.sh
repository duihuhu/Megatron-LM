#!/bin/bash
python ../../tools/preprocess_data.py \
       --input ../codeparrot_data_small.json \
       --output-prefix my-bert \
       --vocab-file bert-large-cased-vocab.txt \
       --tokenizer-type BertWordPieceLowerCase \
       --workers 32 \
       --split-sentences
       