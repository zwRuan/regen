# Evaluating DExperts with chat expert
export PYTHONPATH=/workspace/rzw/proxy-tuning/evaluate/metrics:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1
size=13
python -m eval.gsm.run_eval \
    --data_dir data/eval/gsm \
    --save_dir results/gsm/dexperts-${size}B-alpha2-3iter \
    --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
    --eval_batch_size 1 \
    --alpha 2 \
    --use_chat_format \