import os
import json
import argparse
import logging
import random
import datasets
import pandas as pd
from eval.utils import (
    generate_completions,
    dynamic_import_function,
    load_lm_and_tokenizer,
    load_dexperts_model_and_tokenizer,
    ensure_dir
)
import transformers 
from eval.reward_utils import get_reward_model, get_reward_tokenizer, compute_scores

import debugpy
try:
    # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
    debugpy.listen(("localhost", 16236))
    print("Waiting for debugger attach")
    debugpy.wait_for_client()
except Exception as e:
    pass


def load_json(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data





def main(args):
    random.seed(42)
    ensure_dir(args.save_dir)

    if args.reward_model_name:
        reward_model_name = args.reward_model_name
        reward_tokenizer = get_reward_tokenizer(
            reward_model_name, local_files_only=False
        )
        reward_model = get_reward_model(
            reward_model_name,
            reward_tokenizer,
            "cuda",
            local_files_only=False,
        )

    if args.data_path:
        
        alpaca_eval_data = load_json(os.path.join(args.data_path, f"predictions_all.jsonl"))


    all_reward = []
    best_output = None
    max_reward = float('-inf')
    results = []
    for example in alpaca_eval_data:
        prompt = example["instruction"]
        output = [example["output"]]  
        reward_list = compute_scores(
            prompt,
            output,
            reward_model_name,
            reward_tokenizer,
            reward_model,
        )
        reward = reward_list[0]
        example["reward"] = reward
        all_reward.append(reward)
        results.append(example)
        if reward > max_reward:
            max_reward = reward
    avg_reward = sum(all_reward)/len(all_reward)
    print(avg_reward)
    print(max_reward)
    with open(os.path.join(args.save_dir, f"predictions_rewards.jsonl"), "w") as fout:
        for result in results:
            fout.write(json.dumps(result) + "\n")
        fout.write(json.dumps({"avg_reward": avg_reward, "max_reward": max_reward}) + "\n")
    

    # evaluation_args = {
    #     "model_outputs": model_results,
    #     "annotators_config": "alpaca_eval_gpt4_0314",
    #     "output_path": args.save_dir,
    #     "is_return_instead_of_print": True,
    #     "precomputed_leaderboard": None,
    #     "is_cache_leaderboard": False,
    #     "caching_path": os.path.join(args.save_dir, "alpaca_eval_annotator_cache.json")
    # }

    # if args.reference_path:
    #     evaluation_args["reference_outputs"] = args.reference_path

    # df_leaderboard, annotations = alpaca_farm_evaluate(**evaluation_args)

    # # save to json
    # with open(os.path.join(args.save_dir, "metrics.json"), "w") as fout:
    #     for k in df_leaderboard.to_dict():
    #         df_leaderboard[k] = df_leaderboard[k][model_name]
    #     json.dump(df_leaderboard, fout, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default=None
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/alpaca_farm")
    parser.add_argument(
        "--reward_model_name",
        type=str,
        default=None
    )
    parser.add_argument(
        "--chat_formatting_function",
        type=str,
        default="eval.templates.create_prompt_with_tulu_chat_format",
        help="The function to use to create the chat format. This function will be dynamically imported. Please see examples in `eval/templates.py`."
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation."
    )
    args = parser.parse_args()

    main(args)
