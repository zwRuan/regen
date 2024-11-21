import os
import json
import argparse
import logging
import random
import datasets
import pandas as pd
from alpaca_eval import evaluate as alpaca_farm_evaluate
from eval.utils import (
    generate_completions,
    dynamic_import_function,
    load_lm_and_tokenizer,
    load_dexperts_model_and_tokenizer,
    ensure_dir,
    dexperts_generate_completions
)
import transformers 
def get_templated_prompt(
    prompt: str,
    llm_name: str,
    generation_tokenizer: transformers.PreTrainedTokenizerFast,
) -> str:
    if "Instruct" in llm_name:
        conversation = [
            {"role": "user", "content": prompt},
        ]
        templated_prompt: str = generation_tokenizer.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
    elif any(s in llm_name for s in ["sft10k", "alpaca-7b", "dpo", "ppo", "human"]):
        templated_prompt = f"<s>Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{prompt}\n\n### Response:"
    elif "llama-2" in llm_name.lower():
        templated_prompt = f"<s>[INST]\n{prompt} [/INST]"
    else:
        templated_prompt = generation_tokenizer.bos_token + prompt
    return templated_prompt

def get_posprompt_ID(instruction, orginal_output):
    pos_prompt_template = f'''
        {instruction}
        {orginal_output}

        You are an active observer, skilled at thinking critically. You can refer to the original answer to generate a completely new and unique response, one that differs greatly from the original. Answer the question again.
        {instruction}
        '''
    return pos_prompt_template
    
    
def get_negprompt_ID(instruction, orginal_output):
    neg_prompt_template = f'''
        {instruction}
        {orginal_output}

        You are a passive observer, unwilling to engage in deep thought. Your answer should lack originality and be very close to the original response. Answer the question again. 
        {instruction}
        '''
    return neg_prompt_template


def main(args):
    random.seed(42)
    prefix_outputs = []
    ensure_dir(args.save_dir)
    if args.data_path:
        alpaca_eval_data = pd.read_json(args.data_path).to_dict(orient="records")
    else:
        alpaca_eval_data = datasets.load_dataset("data/eval/alpaca_eval", "alpaca_eval")["eval"]
    for i in range(5):
        if i == 0:
            logging.info("loading data and model...")
            model, tokenizer = load_lm_and_tokenizer(
                model_name_or_path=args.model_name_or_path,
                tokenizer_name_or_path=args.tokenizer_name_or_path,
                load_in_8bit=args.load_in_8bit,
                use_fast_tokenizer=not args.use_slow_tokenizer,
            )
        elif i == 1:
            logging.info("loading dexperts...")
            model, tokenizer = load_dexperts_model_and_tokenizer(
                model_name_or_path=args.model_name_or_path,
                alpha=args.alpha,
                chat_response_prefix="Answer:",
                load_in_8bit=args.load_in_8bit,
                use_fast_tokenizer=not args.use_slow_tokenizer,
            )
        else:
            print(f"正在进行第{i}次迭代, 无需重新加载模型")
            print(f"模型: {model.__class__.__name__}")
            print(f"tokenizer: {tokenizer.__class__.__name__}")
        

        prompts = []
        pos_prompts = []
        neg_prompts = []
        for num_example, example in enumerate(alpaca_eval_data):
            prompt = example["instruction"]
            prompt = get_templated_prompt(prompt, args.model_name_or_path, tokenizer)
            if i != 0:
                # pos_prompts = []
                # neg_prompts = []
                for index in range(1):
                    pos_prompt = get_posprompt_ID(example["instruction"],prefix_outputs[num_example])
                    neg_prompt = get_negprompt_ID(example["instruction"],prefix_outputs[num_example])
                    pos_prompt = get_templated_prompt(pos_prompt, args.model_name_or_path, tokenizer)
                    neg_prompt = get_templated_prompt(neg_prompt, args.model_name_or_path, tokenizer)
                    # pos_prompts.append(pos_prompt)
                    # neg_prompts.append(neg_prompt)
            prompts.append(prompt)
            if i != 0:
                pos_prompts.append(pos_prompt)
                neg_prompts.append(neg_prompt)

        #prompts = prompts[:args.max_examples]

        with open(os.path.join(args.save_dir, "example_prompt.txt"), 'w') as fout:
            fout.write(prompts[0])

        

        stop_sequences = ["\n\nComment:"]  # degenerate stuff for llama 2
        stop_sequences = [tokenizer.encode(" " + x, add_special_tokens=False)[1:] for x in stop_sequences]
        if i == 0:
            outputs = generate_completions(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=args.max_new_tokens,
                batch_size=args.eval_batch_size,
                do_sample=False,
            )
        else:
            outputs = dexperts_generate_completions(
                model=model,
                tokenizer=tokenizer,
                base_prompts=prompts,
                pos_prompts=pos_prompts,
                neg_prompts=neg_prompts,
                max_new_tokens=args.max_new_tokens,
                batch_size=args.eval_batch_size,
                do_sample=True,
            )

        model_results = []
        model_name = os.path.basename(args.save_dir)
        if len(prefix_outputs) ==  0:
            prefix_outputs = [f"Output {i}: " + outputs[index] + "\n" for index in range(len(outputs))]
        else:
            assert len(prefix_outputs) == len(outputs), "prefix_outputs and outputs must have the same length"
            prefix_outputs = [prefix_outputs[index] + f"Output {i}: " + outputs[index] + "\n" for index in range(len(outputs))]
        with open(os.path.join(args.save_dir, f"predictions_{i}.jsonl"), "w") as fout:
            for example, output in zip(alpaca_eval_data, outputs):
                example["output"] = output.strip()
                example["generator"] = model_name
                fout.write(json.dumps(example) + "\n")
                model_results.append(example)
    
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
        "--reference_path",
        type=str,
        default=None,
        help="Path to the reference outputs."
    )
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
        "--max_examples",
        type=int,
        default=None,
        help="The number of instances to evaluate. If not given, we will evaluate all instances."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="If specified, we will load the model to generate the predictions.",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
        help="If specified, we will load the tokenizer from here.",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If given, we will use the slow tokenizer."
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help="Maximum number of new tokens to generate."
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation."
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load model in 8bit mode, which will reduce memory and speed up inference.",
    )
    parser.add_argument(
        "--base_model_name_or_path",
        type=str,
        default='meta-llama/Llama-2-13b-hf',
    )
    parser.add_argument(
        "--expert_model_name_or_path",
        type=str,
        default='meta-llama/Llama-2-7b-chat-hf',
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=None
    )
    parser.add_argument(
        "--use_chat_format",
        action="store_true",
        help="If given, we will use the chat format for the prompts."
    )
    parser.add_argument(
        "--chat_formatting_function",
        type=str,
        default="eval.templates.create_prompt_with_tulu_chat_format",
        help="The function to use to create the chat format. This function will be dynamically imported. Please see examples in `eval/templates.py`."
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
    )
    args = parser.parse_args()

    main(args)
