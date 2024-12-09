from sentence_transformers import SentenceTransformer, util
import numpy as np
import math
import json
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 16236))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass



device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
def get_input_encoding(
    questions: list[str],
    generation_model: transformers.LlamaForCausalLM,
    generation_tokenizer: transformers.PreTrainedTokenizerFast,
) -> transformers.BatchEncoding:
    input_encoding = generation_tokenizer(
        questions, padding=True, add_special_tokens=False, return_tensors="pt"
    ).to(generation_model.device)
    return input_encoding
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


def load_json(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data
# Semantic Entropy Calculation Function
# def compute_semantic_entropy(data, model_name="dunzhang/stella_en_400M_v5", threshold=0.99, min_community_size=1):
#     print("Loading model...")
#     model = SentenceTransformer(model_name, trust_remote_code=True).to(device)
#     text = [d["response"] for d in data]
#     embeddings = model.encode(text, convert_to_tensor=True, show_progress_bar=True, device=device, batch_size=1)
#     #clusters = util.community_detection(embeddings, min_community_size=min_community_size, threshold=threshold)
#     similarities = model.similarity(embeddings, embeddings)
#     print(similarities)
#     # 将对角线上的值设为0
#     similarities.fill_diagonal_(0)
#     print(similarities)
#     # 计算每个句子与其他句子的相似度均值
#     mean_similarities = similarities.sum(dim=1) / (similarities.size(1) - 1)
    
#     # 计算这些均值的均值
#     overall_mean_similarity = mean_similarities.mean().item()
    
#     print("Mean similarities for each sentence:", mean_similarities)
#     print("Overall mean similarity:", overall_mean_similarity)
#     return overall_mean_similarity
    # total_samples = len(data)
    # probabilities = [len(cluster) / total_samples for cluster in clusters]
    # print("Probabilities:", probabilities)
    # entropy = -sum(p * math.log(p) for p in probabilities if p > 0)
    # return entropy, clusters
# Semantic Entropy Calculation Function

def compute_semantic_entropy(model, data, threshold=0.75, min_community_size=1):
    print("Loading model...")
    query_prompt_name = "s2p_query"
    
    text = [d["response"] for d in data]
    embeddings = model.encode(text)
    clusters = util.community_detection(embeddings, min_community_size=min_community_size, threshold=threshold)
    total_samples = len(data)
    probabilities = [len(cluster) / total_samples for cluster in clusters]
    print("Probabilities:", probabilities)
    entropy = -sum(p * math.log(p) for p in probabilities if p > 0)
    return entropy, clusters

# def compute_semantic_entropy(data, model_name="dunzhang/stella_en_400M_v5", threshold=0.99, min_community_size=1):
#     print("Loading model...")
#     model = SentenceTransformer(model_name, trust_remote_code=True).to(device)
#     text = [d["response"] for d in data]
#     embeddings = model.encode(text, convert_to_tensor=True, show_progress_bar=True, device=device, batch_size=32)
#     clusters = util.community_detection(embeddings, min_community_size=min_community_size, threshold=threshold)
#     total_samples = len(data)
#     probabilities = [len(cluster) / total_samples for cluster in clusters]
#     print("Probabilities:", probabilities)
#     entropy = -sum(p * math.log(p) for p in probabilities if p > 0)
#     return entropy, clusters

def save_jsonlines(data, filename):
    with open(filename, 'w') as f:
        for line in data:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')

# Ensure GPU usage


# Load Llama Model and Tokenizer
# model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.pad_token_id = tokenizer.eos_token_id
# tokenizer.padding_side = "left"
# model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
# )
# model.eval()
# for param in model.parameters():
#     param.requires_grad = False    

# Generate Responses Function
def generate_responses(question, num_responses=5, temperature=0.7, top_p=0.9):
    responses = []
    for _ in range(num_responses):
        messages = [
            {"role": "user", "content": question},
        ]
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        #inputs = tokenizer.encode(question, return_tensors="pt").to(device)
        outputs = model.generate(
            input_ids,
            max_new_tokens=2048,
            eos_token_id=terminators,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )
        response = outputs[0][input_ids.shape[-1]:]
        response = tokenizer.decode(response, skip_special_tokens=True)
        responses.append(response)
    return responses

# Strategy Functions
def strategy_one(question):
    params = [
        {"temperature": 0.7, "top_p": 0.9},
        {"temperature": 1.0, "top_p": 0.95},
        {"temperature": 0.5, "top_p": 0.8},
    ]
    
    for param in params:
        entropy_results = []
        for qus in question:
            responses = generate_responses(qus, temperature=param["temperature"], top_p=param["top_p"])
            data = [{"response": resp} for resp in responses]
            entropy, clusters = compute_semantic_entropy(data)
            entropy_results.append({"temperature": param["temperature"], "top_p": param["top_p"], "entropy": entropy})
        avg_entropy = sum([item["entropy"] for item in entropy_results]) / len(entropy_results)
        results = {"temperature": param["temperature"], "top_p": param["top_p"], "avg_entropy": avg_entropy}
        print(results)
        name = f"strategy_one_entropy_{param['top_p']}.jsonl"
        save_jsonlines(results, name)
    return entropy_results

def strategy_two(question):
    responses = generate_responses(question, num_responses=5)
    data = [{"response": resp} for resp in responses]
    entropy, clusters = compute_semantic_entropy(data)
    result = {"strategy": "two", "entropy": entropy, "clusters": clusters}
    save_jsonlines([result], "strategy_two_entropy.jsonl")
    return result

def strategy_three(question):
    summary_pool = []
    responses = []
    current_prompt = question
    for _ in range(5):
        response = generate_responses(current_prompt, num_responses=1)[0]
        summary_prompt = f"Summarize the following response:\n{response}"
        summary = generate_responses(summary_prompt, num_responses=1)[0]
        summary_pool.append(summary)
        responses.append(response)
        current_prompt = f"{question}\nSummaries:\n" + "\n".join(summary_pool)
    data = [{"response": resp} for resp in responses]
    entropy, clusters = compute_semantic_entropy(data)
    save_jsonlines({"responses": responses, "entropy": entropy}, "strategy_three_entropy.jsonl")
    return entropy

def strategy_four(question):
    all_responses = []
    current_prompt = question
    logits_cache = []
    for _ in range(5):
        response = generate_responses(current_prompt, num_responses=1)[0]
        all_responses.append(response)
        current_prompt += f"\nNew Explanation:\n{response}"
        logits_cache.append(response)
    data = [{"response": resp} for resp in all_responses]
    entropy, clusters = compute_semantic_entropy(data)
    save_jsonlines({"responses": all_responses, "entropy": entropy}, "strategy_four_entropy.jsonl")
    return entropy
#results/alpaca_farm/base_logits/alpha_0.5/predictions_all.jsonl
#/data1/rzw/CODE/proxy-tuning/results/alpaca_farm/base_prompt/predictions_all.jsonl
#/data1/rzw/CODE/proxy-tuning/results/alpaca_farm/pos_prompt/predictions_all.jsonl
def test_our_method():
    file_list = [f"/data1/rzw/CODE/proxy-tuning/results/alpaca_farm/pos_prompt/predictions_all.jsonl"]
    model_name = "dunzhang/stella_en_400M_v5"
    model = SentenceTransformer(model_name, trust_remote_code=True).to(device)
    for file in file_list:
        print(file)
        data = load_json(file)
        entropy_results = []
        # 确保数据格式正确，将数据转换为需要的格式
        for i in range(100):
            responses = []
            for j in range(5):
                responses.append({"response": data[100*j+i]["output"]})
                #question = data[100*j+i]["instruction"]
                #print(question)
                #print("#######")
            entropy = compute_semantic_entropy(model,responses,threshold=0.99)
            entropy_results.append({"entropy": entropy})
        # for item in data:
        #     formatted_data.append({
        #         "response": item["output"]  # 从jsonl文件中提取instruction字段
        #     })
        avg_entropy = sum([item["entropy"] for item in entropy_results]) / len(entropy_results)
        print(avg_entropy)
        save_jsonlines({"responses": data, "entropy": entropy_results, "avg_entropy": avg_entropy}, "strategy_base_logits_entropy.jsonl")
        return entropy
test_our_method()



# import pandas as pd
# alpaca_eval_data = pd.read_json("data/eval/alpaca_eval/alpaca_eval_gpt4_baseline.json").to_dict(orient="records")[:100]
# # # Test
# all_strategies_results = []
# prompts = []
# for num_example, example in enumerate(alpaca_eval_data):
#     prompt = example["instruction"]
#     prompts.append(prompt)
# strategy_one(prompts)
    # question = "请解释一下量子纠缠现象。"
    # strategy_results = {
    #     "strategy_one": strategy_one(tokenized_prompts),
    #     "strategy_two": strategy_two(tokenized_prompts),
    #     "strategy_three": strategy_three(tokenized_prompts),
    #     "strategy_four": strategy_four(tokenized_prompts),
    # }
    # all_strategies_results.append(strategy_results)
#print("Strategy Results:", strategy_results)