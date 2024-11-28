# Evaluating DExperts with chat expert
export PYTHONPATH=/workspace/rzw/proxy-tuning/evaluate/metrics:$PYTHONPATH
export HUGGINGFACE_HUB_CACHE=/workspace/CACHE
export CUDA_VISIBLE_DEVICES=0
size=13

# for alpha in 0.3 0.4 0.6 0.7 0.8 1 1.2 1.5 1.8 2 3 5
# do
#     python -m eval.gsm.run_eval_test \
#         --data_dir data/eval/gsm \
#         --save_dir results/gsm/eval/base_logits_${alpha} \
#         --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
#         --eval_batch_size 10 \
#         --alpha ${alpha} \
#         --method None
# done


python -m eval.gsm.run_eval_test \
        --data_dir data/eval/gsm \
        --save_dir results/gsm/eval/test/test \
        --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
        --eval_batch_size 10 \
        --alpha 0 \
        --method None \