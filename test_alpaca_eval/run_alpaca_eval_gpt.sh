OUTPUT_DIR=$1
GPT_ANNOTATOR=$2
# "weighted_alpaca_eval_gpt4_turbo" by default, which points to "gpt-4-1106-preview" at 2024.11;
# "weighted_alpaca_eval_gpt4-06-13"
# "alpaca_eval_gpt4_0613"
# use "weighted_alpaca_eval_gpt-4o-mini-2024-07-18" for lower price

# cal metrics
echo "###### GPT-Annotator: $GPT_ANNOTATOR ######"
mkdir -p $OUTPUT_DIR/$GPT_ANNOTATOR
alpaca_eval --model_outputs $OUTPUT_DIR/file_upload.json \
    --annotators_config $GPT_ANNOTATOR \
    --caching_path $OUTPUT_DIR/$GPT_ANNOTATOR/annotations_cache.json > $OUTPUT_DIR/$GPT_ANNOTATOR/metrics.txt 2>&1
    # set caching_path to none to allow multiple runs at the same time

cat $OUTPUT_DIR/$GPT_ANNOTATOR/metrics.txt
echo "Metrics are saved in $OUTPUT_DIR/$GPT_ANNOTATOR/metrics.txt"