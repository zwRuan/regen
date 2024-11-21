# Evaluating DExperts with chat expert
export PYTHONPATH=/workspace/rzw/proxy-tuning/evaluate/metrics:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
size=13
python -m eval.gsm.run_eval_test \
    --data_dir data/eval/gsm \
    --save_dir results/gsm/eval/do_sample \
    --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
    --eval_batch_size 10 \
    --alpha 0 \
    --do_sample \