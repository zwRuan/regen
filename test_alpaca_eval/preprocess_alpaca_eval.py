import json
from typing import List
def write_jsons(dict_objs: List[dict], file_name: str):
    with open(file_name, "w+", encoding='utf-8') as f:
        for dict_obj in dict_objs:
            f.write(json.dumps(dict_obj, ensure_ascii=False) + "\n")

import datasets
def transforme_to_alpaca_format(output_file):
    eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]

    output = []
    for example in eval_set :
        output.append({
            "instruction": example["instruction"],
            "input": "",
            "output": example["output"]
        })

    write_jsons(output, output_file)
    print(f"Transformed {len(eval_set)} responses to {output_file} in alpaca format")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()
    transforme_to_alpaca_format(args.output_file)