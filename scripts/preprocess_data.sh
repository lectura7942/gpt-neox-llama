python tools/datasets/preprocess_data.py \
    --input ./data/news_data_en_short.jsonl \
    --jsonl-keys text \
    --output-prefix ./data/pretrain_dataset \
    --tokenizer-type HFLlamaTokenizer \
    --append-eod \
    --workers 8