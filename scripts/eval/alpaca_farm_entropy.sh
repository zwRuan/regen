# export CUDA_VISIBLE_DEVICES=4
# python -m eval.alpaca_farm.eval_entropy \
#     --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
#     --save_dir results/alpaca_farm/base_entropy \
#     --data_path data/eval/alpaca_eval/alpaca_eval_gpt4_baseline.json \
#     --eval_batch_size 5 \
#     --alpha 0.5 \
#     --threshold 0.01 \
#     #--do_sample

export CUDA_VISIBLE_DEVICES=5
python -m eval.alpaca_farm.pos_eval \
    --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
    --save_dir results/alpaca_farm/pos_prompt \
    --data_path data/eval/alpaca_eval/alpaca_eval_gpt4_baseline.json \
    --eval_batch_size 5 \
    --do_sample