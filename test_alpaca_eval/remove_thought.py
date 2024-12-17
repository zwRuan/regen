import json
from typing import List
def write_jsons(dict_objs: List[dict], file_name: str):
    with open(file_name, "w+", encoding='utf-8') as f:
        for dict_obj in dict_objs:
            f.write(json.dumps(dict_obj, ensure_ascii=False) + "\n")

def read_jsons(file_name: str):
    dict_objs = []
    with open(file_name, "r") as f:
        for line in f:
            dict_objs.append(json.loads(line))
    return dict_objs

import re
def remove_thought(string, start_token="<THOUGHT>", end_token="</THOUGHT>"):
    pattern = re.escape(start_token) + '.*?' + re.escape(end_token)
    return re.sub(pattern, '', string, flags=re.DOTALL)


def remove_thoughts(data, keys, start_token="<THOUGHT>", end_token="</THOUGHT>"):
    for item in data:
        for key in keys:
            item[key] = remove_thought(item[key], start_token, end_token)
    return data


import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", type=str, required=True)
    parser.add_argument("--out_file", type=str, required=True)
    parser.add_argument("--start_token", type=str, default="<THOUGHT>")
    parser.add_argument("--end_token", type=str, default="</THOUGHT>")
    parser.add_argument("--keys", type=str, nargs="+", default=["predict"])
    args = parser.parse_args()

    data = read_jsons(args.in_file)
    data = remove_thoughts(data, args.keys, args.start_token, args.end_token)
    write_jsons(data, args.out_file)
    print(f"Removed thoughts saved to {args.out_file}")