export CUDA_VISIBLE_DEVICES=0
#METHOD=entropy-juedge


source activate generation
python -m eval.alpaca_farm.run_eval \
    --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
    --data_path /workspace/CODE/regen/alpaca_eval/alpaca_eval.json \
    --save_dir results/test \
    --eval_batch_size 2 \
    --method 1 \
    --weight_method entropy
# for THRESHOLD in 0.001 0.01 0.05 0.1 0.15 0.2 
# do
#     python -m eval.alpaca_farm.run_eval \
#         --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
#         --save_dir results/alpaca_farm_500/threshold_${THRESHOLD}_to_1 \
#         --data_path data/eval/alpaca_eval/alpaca_eval_gpt4_baseline.json \
#         --threshold $THRESHOLD \
#         --eval_batch_size 1 \
#         --method 1 \
#         --use_threshold \
#         --weight_method entropy
# done

#0.25 0.3 0.35 0.4 0.45 0.5
# python -m eval.alpaca_farm.run_eval \
#     --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
#     --save_dir results/alpaca_farm_500/base_entropy \
#     --data_path data/eval/alpaca_eval/alpaca_eval_gpt4_baseline.json \
#     --threshold 0.1 \
#     --eval_batch_size 1 \
#     --method 0 \
#     --weight_method entropy


# for ALPHA in 0.5 1.0
# do
#     python -m eval.alpaca_farm.run_eval \
#         --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
#         --save_dir results/alpaca_farm_500/base_alpha/$ALPHA \
#         --data_path data/eval/alpaca_eval/alpaca_eval_gpt4_baseline.json \
#         --alpha $ALPHA \
#         --eval_batch_size 1 \
#         --method base \
#         --weight_method alpha 
# done





# source activate reward

# for THRESHOLD in 0.1 0.2 0.5
# do
#     DIR=results/alpaca_farm_500/more_${THRESHOLD}_to_1
#     python -m eval.reward.reward_chat \
#         --data_path ${DIR}/predictions_all.jsonl \
#         --save_dir ${DIR}/reward_predictions.jsonl \
#         --reward_model_name Skywork/Skywork-Reward-Llama-3.1-8B \
#         --eval_batch_size 5 
# done




# source activate reward
# DIR=results/alpaca_farm_500/base_entropy
# python -m eval.reward.reward_chat \
#     --data_path ${DIR}/predictions_all.jsonl \
#     --save_dir ${DIR}/reward_predictions.jsonl \
#     --reward_model_name Skywork/Skywork-Reward-Llama-3.1-8B \
#     --eval_batch_size 5 

# source activate reward
# for ALPHA in 0.5 1.0
# do
#     DIR=results/alpaca_farm_500/base_alpha/$ALPHA
#     python -m eval.reward.reward_chat \
#         --data_path ${DIR}/predictions_all.jsonl \
#         --save_dir ${DIR}/reward_predictions.jsonl \
#         --reward_model_name Skywork/Skywork-Reward-Llama-3.1-8B \
#         --eval_batch_size 5 
# done