#export CUDA_VISIBLE_DEVICES=0
#METHOD=entropy-juedge





for THRESHOLD in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 1.0 1.2 1.5 1.7 2.0
do
    python -m eval.alpaca_farm.run_eval \
        --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
        --save_dir results/alpaca_farm/more_${THRESHOLD}_to_2 \
        --data_path data/eval/alpaca_eval/alpaca_eval_gpt4_baseline.json \
        --threshold $THRESHOLD \
        --eval_batch_size 1 \
        --method 2 \
        --weight_method entropy
done


for THRESHOLD in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 1.0 1.2 1.5 1.7 2.0
do
    DIR=results/alpaca_farm/more_${THRESHOLD}_to_2
    python -m eval.alpaca_farm.cal_reward \
        --save_dir $DIR \
        --data_path $DIR \
        --reward_model_name /workspace/rzw/MODEL/ArmoRM-Llama3-8B-v0.1 \
        --eval_batch_size 5
done
