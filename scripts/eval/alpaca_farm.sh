
METHOD=$1
DEVICE=$2
export CUDA_VISIBLE_DEVICES=$DEVICE


for ALPHA in 0.1 0.3 0.5 0.8 1.0 2.0 5.0
do
    python -m eval.alpaca_farm.case_study \
        --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
        --save_dir results/alpaca_farm/$METHOD/alpha_$ALPHA \
        --data_path data/eval/alpaca_eval/alpaca_eval_gpt4_baseline.json \
        --eval_batch_size 5 \
        --alpha $ALPHA \
        --method $METHOD
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