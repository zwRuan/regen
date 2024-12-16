#export CUDA_VISIBLE_DEVICES=0
#METHOD=entropy-juedge




source activate generation
for THRESHOLD in 0.25 0.3 0.35 0.4 0.45 0.5
do
    python -m eval.alpaca_farm.run_eval \
        --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
        --save_dir results/alpaca_farm_500/threshold_${THRESHOLD}_to_1 \
        --data_path data/eval/alpaca_eval/alpaca_eval_gpt4_baseline.json \
        --threshold $THRESHOLD \
        --eval_batch_size 1 \
        --method 1 \
        --use_threshold \
        --weight_method entropy
done

#0.25 0.3 0.35 0.4 0.45 0.5











# for THRESHOLD in 0.5 0.6 1.2 1.5 2.0
# do
#     python -m eval.alpaca_farm.run_eval \
#         --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
#         --save_dir results/alpaca_farm_500/more_${THRESHOLD}_to_2 \
#         --data_path data/eval/alpaca_eval/alpaca_eval_gpt4_baseline.json \
#         --threshold $THRESHOLD \
#         --eval_batch_size 1 \
#         --method 2 \
#         --weight_method entropy
# done


# python -m eval.alpaca_farm.pos_eval \
#     --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
#     --save_dir results/pos_prompt_500/pos_prompt \
#     --data_path data/eval/alpaca_eval/alpaca_eval_gpt4_baseline.json \
#     --eval_batch_size 5 \
#     --pos_or_neg pos

# python -m eval.alpaca_farm.pos_eval \
#     --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
#     --save_dir results/pos_prompt_500/neg_prompt \
#     --data_path data/eval/alpaca_eval/alpaca_eval_gpt4_baseline.json \
#     --eval_batch_size 5 \
#     --pos_or_neg neg

# python -m eval.alpaca_farm.pos_eval \
#     --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
#     --save_dir results/pos_prompt_500/base_prompt \
#     --data_path data/eval/alpaca_eval/alpaca_eval_gpt4_baseline.json \
#     --eval_batch_size 5 \
#     --pos_or_neg base


# source activate reward

# for THRESHOLD in 0.5 0.6 1.2 1.5 2.0
# do
#     DIR=results/alpaca_farm_500/more_${THRESHOLD}_to_2
#     python -m eval.reward.reward_chat \
#         --data_path ${DIR}/predictions_all.jsonl \
#         --save_dir ${DIR}/reward_predictions.jsonl \
#         --reward_model_name Skywork/Skywork-Reward-Llama-3.1-8B \
#         --eval_batch_size 5 
# done

# source activate reward
# DIR=results/pos_prompt_500/pos_prompt
# python -m eval.reward.reward_chat \
#     --data_path ${DIR}/predictions_all.jsonl \
#     --save_dir ${DIR}/reward_predictions.jsonl \
#     --reward_model_name Skywork/Skywork-Reward-Llama-3.1-8B \
#     --eval_batch_size 5 

# source activate reward
# DIR=results/pos_prompt_500/neg_prompt
# python -m eval.reward.reward_chat \
#     --data_path ${DIR}/predictions_all.jsonl \
#     --save_dir ${DIR}/reward_predictions.jsonl \
#     --reward_model_name Skywork/Skywork-Reward-Llama-3.1-8B \
#     --eval_batch_size 5 

# source activate reward
# DIR=results/pos_prompt_500/base_prompt
# python -m eval.reward.reward_chat \
#     --data_path ${DIR}/predictions_all.jsonl \
#     --save_dir ${DIR}/reward_predictions.jsonl \
#     --reward_model_name Skywork/Skywork-Reward-Llama-3.1-8B \
#     --eval_batch_size 5 
