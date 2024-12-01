from copy import deepcopy

from numpy import save
from dataclasses import dataclass
import re
from typing import List
from loguru import logger 
import os
import json
from transformers import AutoTokenizer
from matplotlib import pyplot as plt
import wandb
wandb.login(key="2b8f9ed8fd06fb6076d7dc1b198d1d8de9b5bf23")

from vllm import LLM, SamplingParams


def get_tags_template(user_query):
    tags_template = f"""You are a helpful assistant. Please identify tags of user intentions in the following user query and provide an explanation for each tag. Please respond in the JSON format {{"tag": "str", "explanation": "str"}}.
Query: {user_query} 
Assistant:""" 
    return tags_template


@dataclass
class InferenceConfig:
    temperature: float = 0.0
    top_p: float = 0.9
    skip_special_tokens: bool = True
    model_name_or_path: str = "OFA-Sys/InsTagger"


def extract_tags(input_string):
    pattern = r'"tag":\s*"([^"]*)",\s*"explanation":\s*"([^"]*)"'
    matches = re.findall(pattern, input_string)
    return [{"tag": tag if tag else None, "explanation": explanation if explanation else None} 
            for tag, explanation in matches]

inference_config = InferenceConfig()

def get_tags(data: List[dict]) -> List[dict]:
    instructions = [entry['instruction'] for entry in data]
    prompts = [get_tags_template(instruction) for instruction in instructions]
    responses = parallel_inference_instagger(prompts, max_tokens=512, temperature=0.0, top_p=0.9)
    for i, response in enumerate(responses):
        data[i]["instags"] = response
    return data

# output example 
# [{"tag": "information request", 
# "explanation": "The user is requesting information about a specific topic."}, 
# {"tag": "explanation request",
# "explanation": "The user is requesting an explanation of how something works."}, 
# {"tag": "application request",
# "explanation": "The user is requesting information about the applications of a specific technology."}]



def get_instagger_tags(data: List[dict]) -> List[List[dict]]:
    data = get_tags(data)
    tags_str = [entry["instags"] for entry in data]
    tags_list = []
    for tags in tags_str:
        tags_list.append(extract_tags(tags))
    return tags_list





def get_complexity_diversity(data: List[dict]) -> List[dict]:

    data = get_tags(data)
    tags_str = [entry["instags"] for entry in data]
    tags_list = []
    for tags in tags_str:
        tags_list.append(extract_tags(tags))
    # 计算复杂性
    complexity = [len(tag) for tag in tags_list]
    avg_complexity = sum(complexity) / len(complexity) if len(complexity) > 0 else 0

    # 计算多样性
    try:
        diversity = len(set(tag["tag"] for tags in tags_list for tag in tags))
    except KeyError as e:
        logger.error(f"Missing 'tag' key in one of the tags. Error: {e}")
        diversity = 0

    # 输出调试信息
    logger.debug(f"complexity: {complexity}")
    logger.debug(f"avg_complexity: {avg_complexity}")
    logger.debug(f"diversity: {diversity}")

    return avg_complexity, diversity, len(data)

def calculate_average_tokens(data: List[dict], tokenizer) -> float:
    if 'instruction' not in data[0]:
        raise ValueError("The data does not contain 'instruction' key.")

    token_counts = []
    for entry in data:
        instruction = entry["instruction"]
        tokens = tokenizer.tokenize(instruction)
        token_counts.append(len(tokens))
    avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
    return avg_tokens




def get_infomation_request(data: List[dict]) -> List[dict]:
    tokenizer = AutoTokenizer.from_pretrained(inference_config.model_name_or_path)
    avg_complexity, diversity, num_files = get_complexity_diversity(data)
    avg_tokens = calculate_average_tokens(data, tokenizer)
    return avg_complexity, diversity, avg_tokens
def load_json(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def parallel_inference_instagger(prompts, max_tokens=512, temperature=0.0, top_p=0.9):
    # 初始化vLLM
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    llm = LLM(model=inference_config.model_name_or_path)
    
    # 设置采样参数
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )
    
    # 批量推理
    outputs = llm.generate(prompts, sampling_params)
    
    responses = [output.outputs[0].text for output in outputs]
    return responses







if __name__ == "__main__":



    # files_list = ["/home/admin/research/FANNO/Fanno/compare/self-instruct/data/seeds_new.jsonl",
    #               "/home/admin/research/FANNO/experiment/fanno-human-seed/final_data.jsonl",
    #               "/home/admin/research/FANNO/experiment/ablation_11_25_change3_20000/initial_seed.jsonl"]

    file_list = ["/data1/rzw/CODE/proxy-tuning/results/alpaca_farm/base_alpha_0.5/predictions_all.jsonl"]
    for file in file_list:
        print(file)
        data = load_json(file)
        # 确保数据格式正确，将数据转换为需要的格式
        formatted_data = []
        for item in data:
            formatted_data.append({
                "instruction": item["output"]  # 从jsonl文件中提取instruction字段
            })
        avg_complexity, diversity, avg_tokens = get_infomation_request(formatted_data)
        print(f"Average Complexity: {avg_complexity}")
        print(f"Diversity: {diversity}")
        print(f"Average Tokens: {avg_tokens}")
    #     run = wandb.init(
    #     project="FANNO",
    #     name=file.split("/")[-1],
    #     job_type="training"
    #     )
    #     # 便利所有以ucb开头的文件
    #     ucb_files = [f for f in os.listdir(file) if f.startswith("ucb")]
    #     # 按照ucb文件的顺序进行排序，ucb_aug_0.jsonl, ucb_aug_1.jsonl, ...
    #     ucb_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    #     print(f"ucb_files: {ucb_files}")
    #     for ucb_file in ucb_files:
    #         data_path = f"{file}/{ucb_file}"
    #         data = load_json(data_path)
    #         avg_complexity, diversity, avg_tokens = get_infomation_request(data)

    #         wandb.log({
    #             "average_complexity": avg_complexity,
    #             "diversity": diversity,
    #             "average_tokens": avg_tokens
    #         })

    # # Finish the wandb run
    #     run.finish()


                 # avg-complexity    avg_diversity          avg_token            nv_pair      complexity_c    quality_q  quality_deberta
    # human-175:         2.04            211/175          17.46857142857143
    # fanno-seed:        2.60            674/750          51.023936170212764
    # 175-based-fanno:   3.11           2955/10000         95.94358923428345
    # fanno-based-fanno: 3.01           2976/10000        100.77846810989224

