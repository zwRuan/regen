import datasets

alpaca_eval_data = datasets.load_dataset("/workspace/CODE/regen/data/eval/alpaca_eval", "alpaca_eval")["eval"]
print(alpaca_eval_data)