METHOD=$1

python -m eval.alpaca_farm.eval_entropy \
    --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
    --save_dir results/alpaca_farm/entropy-$METHOD \
    --data_path data/eval/alpaca_eval/alpaca_eval_gpt4_baseline.json \
    --eval_batch_size 5 \
    --method $METHOD

