#export CUDA_VISIBLE_DEVICES=1

# for ALPHA in 2.0 5.0
# do
#     python -m eval.alpaca_farm.run_eval \
#         --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
#         --save_dir results/alpaca_farm/base_alpha/$ALPHA \
#         --data_path data/eval/alpaca_eval/alpaca_eval_gpt4_baseline.json \
#         --alpha $ALPHA \
#         --eval_batch_size 1 \
#         --method base \
#         --weight_method alpha 
# done


python -m eval.alpaca_farm.pos_eval \
    --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
    --save_dir results/pos_prompt/pos_prompt \
    --data_path data/eval/alpaca_eval/alpaca_eval_gpt4_baseline.json \
    --eval_batch_size 5 \
    --pos_or_neg pos

python -m eval.alpaca_farm.pos_eval \
    --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
    --save_dir results/pos_prompt/neg_prompt \
    --data_path data/eval/alpaca_eval/alpaca_eval_gpt4_baseline.json \
    --eval_batch_size 5 \
    --pos_or_neg neg

python -m eval.alpaca_farm.pos_eval \
    --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
    --save_dir results/pos_prompt/base_prompt \
    --data_path data/eval/alpaca_eval/alpaca_eval_gpt4_baseline.json \
    --eval_batch_size 5 \
    --pos_or_neg base
