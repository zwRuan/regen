import nltk
from nltk.translate import bleu_score
from nltk.util import ngrams
#nltk.download('punkt_tab')

import json
def load_json(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def calculate_self_bleu(sentences, n=2):
    """
    计算给定句子列表的 self-BLEU。
    :param sentences: 句子列表
    :param n: n-gram 的数量，默认为4
    :return: self-BLEU 值
    """
    references = [nltk.word_tokenize(sentence) for sentence in sentences]
    hypotheses = [references[i] for i in range(len(references))]
    bleu_scores = []

    for i in range(len(hypotheses)):
        # 排除自身句子
        ref = [references[j] for j in range(len(references)) if j != i]
        bleu = bleu_score.sentence_bleu(ref, hypotheses[i], weights=(1/n,)*n)
        bleu_scores.append(bleu)
    
    self_bleu = sum(bleu_scores) / len(bleu_scores)
    return self_bleu

def test_our_method(filename,n=2):
    file_list = [filename]
    for file in file_list:
        print(file)
        data = load_json(file)
        scores = []
        # 确保数据格式正确，将数据转换为需要的格式
        for i in range(10):
            responses = []
            for j in range(5):
                responses.append(data[10*j+i]["output"])
            score = calculate_self_bleu(responses,n=n)
            scores.append(score)
        print("n:",n)
        print(len(scores))
        avg_score = sum(scores) / len(scores)
        print(avg_score)
        return avg_score
#

if __name__ == '__main__':
    filename = "results/alpaca_farm/base_alpha0.5/predictions_all.jsonl"
    test_our_method(filename,n=2)
    test_our_method(filename,n=3)
    test_our_method(filename,n=4)

#print(f"Self-BLEU: {self_bleu_value:.4f}")

