export CUDA_VISIBLE_DEVICES=0
source activate reward
DIR=results/alpaca_farm/base_entropy
#NUM=5000



python -m eval.reward.reward_chat \
    --data_path ${DIR}/predictions_all.jsonl \
    --save_dir ${DIR}/reward_predictions.jsonl \
    --reward_model_name Skywork/Skywork-Reward-Llama-3.1-8B \
    --eval_batch_size 5 



# for ALPHA in 0.1 0.3 0.4 0.5 0.6 0.7 0.8 1.0 2.0 3.0 4.0 5.0

# do
#     python -m eval.reward.reward_chat \
#         --data_path ${DIR}/${ALPHA}/predictions_all.jsonl \
#         --save_dir ${DIR}/${ALPHA}/reward_predictions.jsonl \
#         --reward_model_name Skywork/Skywork-Reward-Llama-3.1-8B \
#         --eval_batch_size 5 
# done






# DIR=results/alpaca_farm
# #NUM=5000
# for THRESHOLD in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 1.0

# do
#     python -m eval.reward.reward_chat \
#         --data_path ${DIR}/more_${THRESHOLD}_to_1/predictions_all.jsonl \
#         --save_dir ${DIR}/more_${THRESHOLD}_to_1/reward_predictions.jsonl \
#         --reward_model_name Skywork/Skywork-Reward-Llama-3.1-8B \
#         --eval_batch_size 5 
# done



# DIR=results/alpaca_farm
# #NUM=5000
# for THRESHOLD in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 1.0 1.2 1.5 1.7 2.0

# do
#     python -m eval.reward.reward_chat \
#         --data_path ${DIR}/more_${THRESHOLD}_to_2/predictions_all.jsonl \
#         --save_dir ${DIR}/more_${THRESHOLD}_to_2/reward_predictions.jsonl \
#         --reward_model_name Skywork/Skywork-Reward-Llama-3.1-8B \
#         --eval_batch_size 5 
# done