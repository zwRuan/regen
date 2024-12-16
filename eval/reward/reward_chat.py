import torch

from transformers import AutoModelForSequenceClassification, AutoTokenizer



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

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 16236))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass


def load_json(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def get_llama3_prompt(prompt, content):
    prompt = f"<|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>"
    return prompt

def get_question_prompt(prompt):
    prompt = f"<|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    return prompt

def main(args):
    random.seed(42)
    #ensure_dir(args.save_dir)
    # Load model and tokenizer
    device = "cuda"
    eot_token = "[MORE TOKENS ARE TRUCATED]"
    model_name = args.reward_model_name
    rm = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="flash_attention_2",
        num_labels=1,
    )
    rm_tokenizer = AutoTokenizer.from_pretrained(model_name)
    if args.data_path:
        
        alpaca_eval_data = load_json(args.data_path)


    all_reward = []
    best_output = None
    max_reward = float('-inf')
    results = []
    print("NUM_TOKENS: ", args.num_tokens)
    for i,example in enumerate(alpaca_eval_data):
        prompt = example["instruction"]
        response = example["output"]
        #conv = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
        #conv_formatted = rm_tokenizer.apply_chat_template(conv, tokenize=False)
        conv_formatted = get_llama3_prompt(prompt,response)
        #question_formatted = get_question_prompt(prompt)
        conv_tokenized = rm_tokenizer(conv_formatted, return_tensors="pt").to(device)
        #question_tokenized = rm_tokenizer(question_formatted, return_tensors="pt").to(device)
        #eot_tokenized = rm_tokenizer(eot_token)
        #len_question = question_tokenized["input_ids"].shape[1]
        #len_all = conv_tokenized["input_ids"].shape[1]
        #len_output = len_all - len_question
        #num_tokens = args.num_tokens


        # input_ids = conv_tokenized["input_ids"]
        # input_ids = torch.cat((input_ids, eot_tokenized["input_ids"]), dim=1)
        # attention_mask = torch.cat((conv_tokenized["attention_mask"][:,:len_question+num_tokens],eot_tokenized["attention_mask"]), dim=1)
        # #attention_mask = conv_tokenized["attention_mask"][:,:len_question+num_tokens]
        # output_input_ids = conv_tokenized["input_ids"][:,len_question:len_question+num_tokens]

        #input_ids = torch.cat((question_tokenized["input_ids"], conv_tokenized["input_ids"][:,len_question+300:len_question+500]), dim=1)
        #attention_mask = torch.cat((question_tokenized["attention_mask"], conv_tokenized["attention_mask"][:,len_question+300:len_question+500]), dim=1)


        # Get the reward scores
        with torch.no_grad():
            reward = rm(**conv_tokenized).logits[0][0].item()
            #reward = rm(**conv_tokenized).logits[0][0].item()
        #print(f"Score for response 1: {reward}")
        #example['output'] = rm_tokenizer.decode(input_ids[0], skip_special_tokens=True)
        example["reward"] = round(reward, 2)
        print(f"R{i}: {example['reward']}")
        all_reward.append(reward)
        results.append(example)
        if reward > max_reward:
            max_reward = reward
    avg_reward = sum(all_reward)/len(all_reward)
    print(args.data_path)
    print("data_num:", len(all_reward))
    print("avg_reward:", avg_reward)
    print("max_reward:", max_reward)
    with open(args.save_dir, "w") as fout:
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
    parser.add_argument(
        "--num_tokens",
        type=int,
        default=10000,
        help="Batch size for evaluation."
    )
    args = parser.parse_args()

    main(args)

















# Output:
# Score for response 1: 12.625
# Score for response 2: -15.25
