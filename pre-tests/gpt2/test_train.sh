#!/bin/bash
python ../../tools/preprocess_data.py \
       --input ../codeparrot_data_small.json \
       --output-prefix codeparrot \
       --vocab-file gpt2-vocab.json \
       --tokenizer-type GPT2BPETokenizer \
       --merge-file gpt2-merges.txt \
       --json-keys content \
       --workers 32 \
       --append-eod