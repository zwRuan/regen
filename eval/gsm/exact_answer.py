import argparse
import os
import re
import json
import random
import evaluate

import transformers 
# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 16236))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass




def trim_output(output):
    instruction_prefix = "Answer the following question"
    question_prefix = 'Question:'
    comment_prefix = 'Comment:'  # for some reason, Llama 13B likes to generate these comments indefinitely

    for prefix in [instruction_prefix, question_prefix, comment_prefix]:
        if prefix in output:
            output = output.split(prefix)[0]

    return output

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


def get_input_encoding(
    questions: list[str],
    generation_model: transformers.LlamaForCausalLM,
    generation_tokenizer: transformers.PreTrainedTokenizerFast,
) -> transformers.BatchEncoding:
    input_encoding = generation_tokenizer(
        questions, padding=True, add_special_tokens=False, return_tensors="pt"
    ).to(generation_model.device)
    return input_encoding


def main():
    random.seed(42)
    print("Loading data...")
    test_data = []
    with open("/data1/rzw/CODE/proxy-tuning/data/downloads/gsm8k.json") as fin:
        for line in fin:
            example = json.loads(line)
            example = json.loads(line)
            temp_ans = example["messages"][1]["content"].split("####")[1]
            temp_ans = int(temp_ans.replace(',', ''))
            test_data.append({
                "question": example["messages"][0]["content"],
                "original_response": example["messages"][1]["content"],
                "extract_answer": temp_ans,
            })
    with open("/data1/rzw/CODE/proxy-tuning/data/exact/gsm8k_exact.json", "w") as fout:
        for data in test_data:
            fout.write(json.dumps(data) + "\n")

if __name__ == "__main__":
    
    main()