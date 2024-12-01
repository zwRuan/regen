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
FIRST_N_TOKENS=$1
DEVICE=$2
export CUDA_VISIBLE_DEVICES=$DEVICE
# Evaluating Llama 2 chat
size=13

for FIRST_N_TOKEN in $FIRST_N_TOKENS
do
    python -m eval.alpaca_farm.case_study \
        --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
        --save_dir results/alpaca_farm/base_alpha_0.5_first$FIRST_N_TOKEN \
        --data_path data/eval/alpaca_eval/alpaca_eval_gpt4_baseline.json \
        --eval_batch_size 5 \
        --alpha 0.5 \
        --first_n_tokens $FIRST_N_TOKEN
done
# for THESHOLD in 0.1 0.2 0.3 0.4 0.5
# do
#     python -m eval.alpaca_farm.case_study \
#         --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
#         --save_dir results/alpaca_farm/THESHOLD_$THESHOLD \
#         --data_path data/eval/alpaca_eval/alpaca_eval_gpt4_baseline.json \
#         --eval_batch_size 1 \
#         --alpha 0.5 \
#         --threshold $THESHOLD \
#         #--do_sample
# done
# python -m eval.alpaca_farm.case_study \
#         --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
#         --save_dir results/alpaca_farm/base_entropy \
#         --data_path data/eval/alpaca_eval/test.json \
#         --eval_batch_size 1 \
#         --alpha 0.5 \
#         --threshold 0.01 \
#         #--do_sample