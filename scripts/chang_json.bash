METHOD=pos_neg_log_softmax

for ALPHA in 0.1 0.3 0.5 0.8 1.0 2.0 5.0
do
    python ./change_json.py \
        --input_dir "results/alpaca_farm/$METHOD/alpha_$ALPHA" \
        --output_file "results/alpaca_farm/$METHOD/alpha_$ALPHA/predictions_all.jsonl"
done
