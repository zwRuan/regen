#export CUDA_VISIBLE_DEVICES=0
#METHOD=entropy-juedge



python -m eval.alpaca_farm.run_eval \
    --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
    --save_dir results/alpaca_farm/base_entropy \
    --data_path data/eval/alpaca_eval/alpaca_eval_gpt4_baseline.json \
    --alpha 0.5 \
    --eval_batch_size 1 \
    --method base \
    --weight_method entropy




# python -m eval.alpaca_farm.run_eval \
#     --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
#     --save_dir results/alpaca_farm/base_base-entropy \
#     --data_path data/eval/alpaca_eval/alpaca_eval_gpt4_baseline.json \
#     --alpha 0.5 \
#     --eval_batch_size 1 \
#     --method base \
#     --weight_method entropy 

