import json
def write_json(dict_objs, file_name):
    with open(file_name, "w+", encoding='utf-8') as f:
        json.dump(dict_objs, f, indent=4, ensure_ascii=False)

def read_jsons(file_name: str):
    dict_objs = []
    with open(file_name, "r") as f:
        for line in f:
            dict_objs.append(json.loads(line))
    return dict_objs

import datasets
from datetime import datetime
def format_results(input_response_data, out_file):
    """
    Format the llama factory results to a valid evaluation format.

    output format:
    https://github.com/google-research/google-research/tree/master/instruction_following_eval#how-to-run

    # {"prompt": "Write a 300+ word summary ...", "response": "PUT YOUR MODEL RESPONSE HERE"}
    # {"prompt": "I am planning a trip to ...", "response": "PUT YOUR MODEL RESPONSE HERE"}
    """
    input_data = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]
    results = read_jsons(input_response_data)
    time_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    formatted_results = []
    for data, result in zip(input_data, results):
        formatted_result = {
            "instruction": data["instruction"],
            "output": result["predict"],
            "generator": "eval_"+time_stamp,
            "dataset": data["dataset"],
            "datasplit": "eval",
        }
        formatted_results.append(formatted_result)
    write_json(formatted_results, out_file)


if __name__ == "__main__":
    import os
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_response_data", type=str, required=True)
    parser.add_argument("--out_file", type=str, default=None)
    args = parser.parse_args()

    if args.out_file is None:
        args.out_file = os.path.join(os.path.dirname(args.input_response_data), "file_upload.json")

    format_results(args.input_response_data, args.out_file)