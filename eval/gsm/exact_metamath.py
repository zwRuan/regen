import argparse
import os
import re
import json
import random
import evaluate
from fraction import Fraction
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

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False


import re
def parse_results(completion):
    remove_dollar = lambda s: s.replace('$', '').strip()
    remove_boxed = lambda s: re.sub(r'\\boxed{(.*?)}', r'\1', s).strip()

    split_ans = completion.split('The answer is:')
    if len(split_ans) <= 1:
        split_ans = completion.split('the answer is:')
    if len(split_ans) <= 1:
        split_ans = completion.split('The final answer is:')
    if len(split_ans) <= 1:
        split_ans = completion.split('the final answer is:')
    if len(split_ans) <= 1:
        if re.findall(r'(\\boxed{.*?})', completion):
            extract_last_boxed = lambda completion: re.findall(r'(\\boxed{.*?})', completion)[-1]
            split_ans = [extract_last_boxed(completion)]
    if len(split_ans) <= 1:
        split_ans = completion.split('####')

    if len(split_ans) > 1:
        ans = split_ans[-1]
        extract_ans_temp = ans.split('.\n')[0].strip()
        extract_ans_temp = remove_dollar(extract_ans_temp)
        extract_ans_temp = remove_boxed(extract_ans_temp)
        if len(extract_ans_temp) > 0 and extract_ans_temp[-1] == '.':
            extract_ans = extract_ans_temp[0:-1]
        else:
            extract_ans = extract_ans_temp
        extract_ans = extract_ans.strip()

        # extract number from string
        match = re.search(r'[\-+]?\d*[\.,/]?\d+', extract_ans)
        if match:
            if '/' in match.group():
                denominator = match.group().split('/')[1]
                numerator = match.group().split('/')[0]
                if is_number(denominator) == True and is_number(numerator) == True:
                    if denominator == '0':
                        return round(float(numerator.replace(',', '')))
                    else:
                        frac = Fraction(match.group().replace(',', ''))
                        num_numerator = frac.numerator
                        num_denominator = frac.denominator
                        return round(float(num_numerator / num_denominator))
                else:
                    return None
            else:
                if float(match.group().replace(',', '')) == float('inf'):
                    return None
                return round(float(match.group().replace(',', '')))
        else:
            return None
    else:
        return None


def main():
    random.seed(42)
    print("Loading data...")
    test_data = []
    with open("/data1/rzw/CODE/proxy-tuning/data/downloads/MetaMathQA/MetaMathQA-395K.json") as fin:
        data = json.load(fin)  # 一次性加载整个文件
        for example in data:
            temp_ans = parse_results(example["response"])
            test_data.append({
                "question": example["query"],
                "original_response": example["response"],
                "extract_answer": temp_ans,
            })
    with open("/data1/rzw/CODE/proxy-tuning/data/exact/metamath_exact.json", "w") as fout:
        for data in test_data:
            fout.write(json.dumps(data) + "\n")

if __name__ == "__main__":
    
    main()