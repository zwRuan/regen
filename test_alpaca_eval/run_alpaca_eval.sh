MODEL_NAME_OR_PATH=$1
TEMPLATE=$2
OUTPUT_DIR=$3
EVAL_BATCH_SIZE=$4
GPT_ANNOTATOR=$5
# "weighted_alpaca_eval_gpt4_turbo" by default, which points to "gpt-4-1106-preview" at 2024.11;
# use "weighted_alpaca_eval_gpt-4o-mini-2024-07-18" for lower price

# remove / at the end of OUTPUT_DIR
if [[ $OUTPUT_DIR == */ ]]; then
    OUTPUT_DIR=${OUTPUT_DIR::-1}
fi

# generate predictions
llamafactory-cli train \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --template $TEMPLATE \
    --output_dir $OUTPUT_DIR \
    --per_device_eval_batch_size $EVAL_BATCH_SIZE \
    --stage sft \
    --do_predict True \
    --finetuning_type full \
    --eval_dataset alpaca_eval \
    --cutoff_len 2048 \
    --overwrite_cache True \
    --overwrite_cache True \
    --preprocessing_num_workers 16 \
    --dataset_dir data \
    --predict_with_generate True \
    --report_to none \
    --seed 42 \
    --do_sample False \
    --temperature 0 \
    --top_k 0 \
    --top_p 0 \
    --max_length 2048 \
    --max_new_tokens 2048

# remove all_results.json, predict_results.json, and trainer_log.jsonl
rm $OUTPUT_DIR/all_results.json
rm $OUTPUT_DIR/predict_results.json
rm $OUTPUT_DIR/trainer_log.jsonl

# remove <THOUGHT> </THOUGHT>
python eval/alpaca_eval/remove_thought.py \
    --in_file $OUTPUT_DIR/generated_predictions.jsonl \
    --out_file $OUTPUT_DIR/generated_predictions_clean.jsonl
echo "Cleaned predictions are saved in $OUTPUT_DIR/generated_predictions_clean.jsonl"

# convert to required format
python eval/alpaca_eval/format_results.py \
    --input_response_data $OUTPUT_DIR/generated_predictions_clean.jsonl \
    --out_file $OUTPUT_DIR/file_upload.json
echo "Results are saved in $OUTPUT_DIR/file_upload.json"

# generate html
python eval/alpaca_eval/generate_html.py \
    --in_file $OUTPUT_DIR/generated_predictions.jsonl \
    --output_dir $OUTPUT_DIR
echo "HTML is saved in $OUTPUT_DIR/$GPT_ANNOTATOR/case_study.html"

# cal metrics
ts bash eval/alpaca_eval/run_alpaca_eval_gpt.sh $OUTPUT_DIR $GPT_ANNOTATOR