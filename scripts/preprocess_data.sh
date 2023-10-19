python tools/datasets/preprocess_data.py \
    --input ./data/news_data_2gb.jsonl \
    --jsonl-keys text \
    --output-prefix ./data/2gb \
    --tokenizer-type HFLlamaTokenizer \
    --append-eod \
    --num-docs 897851 \
    --workers 16