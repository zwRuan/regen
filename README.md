# regen

pip install flash-attn --no-build-isolation

train
bash scripts/eval/alpaca_farm_entropy.sh

eval tag
python eval/diversity/diversity.py