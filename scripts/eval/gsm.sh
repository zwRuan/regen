# Evaluating DExperts with chat expert
export PYTHONPATH=/workspace/rzw/proxy-tuning/evaluate/metrics:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1
size=13
echo "Results dir: results/gsm/dexperts-${size}B"
python -m eval.gsm.run_eval \
    --data_dir data/eval/gsm/ \
    --save_dir results/gsm/dexperts-${size}B \
    --regen_model_name_or_path meta-math/MetaMath-7B-V1.0 \
    --eval_batch_size 20
