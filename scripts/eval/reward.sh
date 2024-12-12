export CUDA_VISIBLE_DEVICES=0
# Evaluating Llama 2 chat
size=13
DIR=results/pos_prompt/pos_prompt/
python -m eval.alpaca_farm.cal_reward \
    --save_dir $DIR \
    --data_path $DIR \
    --reward_model_name RLHFlow/ArmoRM-Llama3-8B-v0.1 \
    --eval_batch_size 5 \