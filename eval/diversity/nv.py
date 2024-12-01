from dataclasses import dataclass
from typing import List, Dict
import tqdm
import benepar, spacy
import ray
benepar.download('benepar_en3')
import json
import time
# 直接设置配置参数，替代原来的 NVConfig
GPU_NUM = 1  # 设置要使用的 GPU 数量

class CustomParser():
    def __init__(self):
        self.nlp = spacy.load('en_core_web_md')
        if spacy.__version__.startswith('2'):
            self.nlp.add_pipe(benepar.BeneparComponent("benepar_en3"))
        else:
            self.nlp.add_pipe("benepar", config={"model": "benepar_en3"})

    def parse(self, text):
        if '\n' in text:
            text = text.replace('\n', ' ')
        while '  ' in text:
            text = text.replace('  ', ' ')
        doc = self.nlp(text.strip())
        return doc


    def parse_map(self, text):
        doc = self.parse(text)
        words_map = {}
        for token in doc:
            if token.dep_ not in words_map:
                words_map[token.dep_] = []
            words_map[token.dep_].append(token.text)
        return words_map

    def parse_verb_nouns_pair(self, text):
        doc = self.parse(text)
        pairs = []
        for token in doc:
            found = False
            if token.pos_ == "VERB":
                for child in token.children:
                    if child.pos_ == "NOUN":
                        pairs.append((token.lemma_, child.text))
                        found = True
                        break  # Stop searching for nouns after finding one
                if found:
                    break
        return pairs


@ray.remote(num_gpus=1)
def process_batch(batch: List[Dict]) -> List:
    parser = CustomParser() 
    pairs = []
    for d in tqdm.tqdm(batch):
        try:
            pair = parser.parse_verb_nouns_pair(d['output'])
            pairs.extend(pair)
        except Exception as e:
            print(f"Error processing entry: {e}")
            continue
    return pairs


def get_nv_analysis(data, num_gpus):
    ray.init()
    batch_size = len(data) // GPU_NUM  # 使用直接定义的 GPU_NUM
    data_batches = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
    results = ray.get([process_batch.remote(batch) for batch in data_batches])
    output_data = [pair for result in results for pair in result]
    ray.shutdown()
    return output_data


def load_jsonlines(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

if __name__ == "__main__":
    #first_tokens = [800, 700, 600, 500, 450, 350, 300, 250, 200]
    #for first_token in first_tokens:
    file_path = f"/data1/rzw/CODE/proxy-tuning/results/alpaca_farm/base_entropy/predictions_all.jsonl"
    
    data = load_jsonlines(file_path)
    
    pairs = get_nv_analysis(data, GPU_NUM)  # 使用直接定义的 GPU_NUM
    #print(f"first_tokens:{first_token}")
    print(f"Total pairs found: {len(pairs)}")
    print(f"Unique pairs found: {len(set(pairs))}")
    
    output_file = f"nv_pairs_analysis_pos_do_sample.txt"
    with open(output_file, 'w') as f:
        f.write(f"Total pairs: {len(pairs)}\n")
        f.write(f"Unique pairs: {len(set(pairs))}\n")
        f.write("\nAll unique pairs:\n")
        for pair in sorted(set(pairs)):
            f.write(f"{pair}\n")

    time.sleep(10)



























                