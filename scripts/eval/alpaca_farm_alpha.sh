#export CUDA_VISIBLE_DEVICES=2
#METHOD=entropy-juedge


for ALPHA in 0.1 0.3 0.4 0.5 0.6 0.7 0.8 1.0 2.0 3.0 4.0 5.0
do
    python -m eval.alpaca_farm.run_eval \
        --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
        --save_dir results/alpaca_farm/base_alpha/$ALPHA \
        --data_path data/eval/alpaca_eval/alpaca_eval_gpt4_baseline.json \
        --alpha $ALPHA \
        --eval_batch_size 1 \
        --method base \
        --weight_method alpha 
done






# python -m eval.alpaca_farm.run_eval \
#     --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
#     --save_dir results/alpaca_farm/base_base-entropy \
#     --data_path data/eval/alpaca_eval/alpaca_eval_gpt4_baseline.json \
#     --alpha 0.5 \
#     --eval_batch_size 1 \
#     --method base \
#     --weight_method entropy 

