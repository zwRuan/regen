# 打开一个新的文件用于写入合并后的内容
# first_tokens = [350, 300, 250, 200, 150, 100, 50]
# for first_token in first_tokens:
with open(f'/data1/rzw/CODE/proxy-tuning/results/alpaca_farm/base_entropy/predictions_all.jsonl', 'w') as outfile:
    # 逐个读取每个文件的内容并写入到新的文件中
    for i in range(5):
        filename = f'/data1/rzw/CODE/proxy-tuning/results/alpaca_farm/base_entropy/predictions_{i}.jsonl'
        with open(filename, 'r') as infile:
            outfile.write(infile.read())