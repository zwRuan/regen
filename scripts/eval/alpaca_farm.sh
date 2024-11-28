# Evaluating DExperts
# size=13
# python -m eval.alpaca_farm.run_eval \
#     --base_model_name_or_path meta-llama/Llama-2-${size}b-hf \
#     --expert_model_name_or_path meta-llama/Llama-2-7b-chat-hf \
#     --save_dir results/alpaca_farm/dexperts-${size}B \
#     --eval_batch_size 8


# Evaluating Llama 2
# size=13
# python -m eval.alpaca_farm.run_eval \
#     --model_name_or_path meta-llama/Llama-2-${size}b-hf \
#     --save_dir results/alpaca_farm/llama2-${size}B \
#     --eval_batch_size 4

export CUDA_VISIBLE_DEVICES=0
# Evaluating Llama 2 chat
size=13
python -m eval.alpaca_farm.case_study \
    --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
    --save_dir results/alpaca_farm/base_alpha_0.5 \
    --data_path data/eval/alpaca_eval/alpaca_eval_gpt4_baseline.json \
    --eval_batch_size 1 \
    --alpha 0.5 \
