import argparse
import os
import re
import json
import random
import evaluate
from eval.utils import (
    generate_completions,
    load_lm_and_tokenizer,
    load_dexperts_model_and_tokenizer,
    dynamic_import_function,
    ensure_dir,
    dexperts_generate_completions
)
import transformers 
import debugpy
try:
    # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
    debugpy.listen(("localhost", 16236))
    print("Waiting for debugger attach")
    debugpy.wait_for_client()
except Exception as e:
    pass

exact_match = evaluate.load("evaluate/metrics/exact_match/exact_match.py")


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


def get_posprompt_template(instruction, ourginal_output):
    pos_prompt_template = f'''
        You are an active observer, skilled at thinking critically. You can refer to the original answer to generate a completely new and unique response, one that differs greatly from the original. Your new answer should showcase a fresh perspective and have significant differences in content and structure.

        Question
        {instruction}
        Original output:
        {ourginal_output}
        Your output:
        '''
    return pos_prompt_template
    
    
def get_negprompt_template(instruction, ourginal_output):
    neg_prompt_template = f'''
        You are a passive observer, unwilling to engage in deep thought. When referring to the original answer, your response should be extremely similar, with minimal deviation. Your answer should lack originality and be very close to the original response.

        Question:
        {instruction}
        Original output:
        {ourginal_output}
        Your output:
        '''
    return neg_prompt_template

def get_posprompt_ID(instruction, orginal_output):
    pos_prompt_template = f'''
        Question:\n{instruction}
        {orginal_output}

        You are an active observer, skilled at thinking critically. You can refer to the original answer to generate a completely new and unique response, one that differs greatly from the original. Answer the question again.
        Question:\n{instruction}
        '''
    return pos_prompt_template
    
    
def get_negprompt_ID(instruction, orginal_output):
    neg_prompt_template = f'''
        Question:\n{instruction}
        {orginal_output}

        You are a passive observer, unwilling to engage in deep thought. Your answer should lack originality and be very close to the original response. Answer the question again. 
        Question:\n{instruction}
        '''
    return neg_prompt_template




def main(args):
    random.seed(42)
    prefix_outputs = []
    print("Loading data...")
    test_data = []
    with open(os.path.join(args.data_dir, "test_error.jsonl")) as fin:
        for line in fin:
            example = json.loads(line)
            test_data.append({
                "question": example["question"],
                "answer": example["answer"].split("####")[1].strip()
            })
            # test_data.append({
            #     "question": example["question"],
            #     "answer": example["answer"],
            # })

    # some numbers are in the `x,xxx` format, and we want to remove the comma
    for example in test_data:
        example["answer"] = re.sub(r"(\d),(\d)", r"\1\2", example["answer"])
        assert float(example["answer"]), f"answer is not a valid number: {example['answer']}"

    if args.max_examples and len(test_data) > args.max_examples:
        test_data = random.sample(test_data, args.max_examples)
    test_data = test_data
    ensure_dir(args.save_dir)
    all_predictions = {}
    for num_predictions in range(5):
        all_predictions[num_predictions] = []
    for j in range(1):
        for i in range(5):
            if i == 0:
                print("Loading model and tokenizer...")
                model, tokenizer = load_lm_and_tokenizer(
                    model_name_or_path=args.model_name_or_path,
                    tokenizer_name_or_path=args.tokenizer_name_or_path,
                    load_in_8bit=args.load_in_8bit,
                    use_fast_tokenizer=not args.use_slow_tokenizer,
                )
            elif i == 1:
                if args.do_sample:
                    print(f"do_sample, 无需重新加载模型")
                else:
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

            prompt_prefix = "Answer the following question.\n\n"
            prompts = []
            # if i != 0:
            pos_prompts = []
            neg_prompts = []
            #chat_formatting_function = dynamic_import_function(args.chat_formatting_function) if args.use_chat_format else None
            for num_example, example in enumerate(test_data):
                prompt = prompt_prefix + "Question: " + example["question"].strip()
                prompt = get_templated_prompt(prompt, args.model_name_or_path, tokenizer)
                # if prompt[-1] in ["\n", " "]:
                #     prompt += "Answer:"
                # else:
                #     prompt += " Answer:"
                if i != 0:
                    # pos_prompts = []
                    # neg_prompts = []
                    for index in range(1):
                        pos_prompt = get_posprompt_ID(example["question"].strip(),prefix_outputs[num_example])
                        neg_prompt = get_negprompt_ID(example["question"].strip(),prefix_outputs[num_example])
                        pos_prompt = get_templated_prompt(pos_prompt, args.model_name_or_path, tokenizer)
                        neg_prompt = get_templated_prompt(neg_prompt, args.model_name_or_path, tokenizer)
                        # pos_prompts.append(pos_prompt)
                        # neg_prompts.append(neg_prompt)
                prompts.append(prompt)
                if i != 0:
                    pos_prompts.append(pos_prompt)
                    neg_prompts.append(neg_prompt)

            # with open(os.path.join(args.save_dir, "example_prompt.txt"), 'w') as fout:
            #     fout.write(prompts[0][0]['content'])

            
            if i == 0:
                outputs = generate_completions(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=prompts,
                    max_new_tokens=512,
                    batch_size=args.eval_batch_size,
                    pad_token_id = tokenizer.eos_token_id,
                    do_sample=False,
                )
            else:
                if args.do_sample:
                    outputs = generate_completions(
                        model=model,
                        tokenizer=tokenizer,
                        prompts=prompts,
                        max_new_tokens=512,
                        batch_size=args.eval_batch_size,
                        pad_token_id = tokenizer.eos_token_id,
                        do_sample=True,
                    )
                else:
                    outputs = dexperts_generate_completions(
                        model=model,
                        tokenizer=tokenizer,
                        base_prompts=prompts,
                        pos_prompts=pos_prompts,
                        neg_prompts=neg_prompts,
                        method=args.method,
                        max_new_tokens=512,
                        batch_size=args.eval_batch_size,
                        pad_token_id = tokenizer.eos_token_id,
                        do_sample=True,
                    )
            outputs = [trim_output(o) for o in outputs]
            if len(prefix_outputs) ==  0:
                prefix_outputs = [f"Output {i}: " + outputs[index] + "\n" for index in range(len(outputs))]
            else:
                assert len(prefix_outputs) == len(outputs), "prefix_outputs and outputs must have the same length"
                prefix_outputs = [prefix_outputs[index] + f"Output {i}: " + outputs[index] + "\n" for index in range(len(outputs))]
            predictions = []
            for output in outputs:
                # replace numbers like `x,xxx` with `xxxx`
                output = re.sub(r"(\d),(\d)", r"\1\2", output)
                numbers = re.findall(r"[-+]?\d*\.\d+|\d+", output)
                if numbers:
                    predictions.append(numbers[-1])
                else:
                    predictions.append(output)
            all_predictions[i] = predictions
            print("Calculating accuracy...")
            targets = [example["answer"] for example in test_data]

            em_score = exact_match.compute(predictions=predictions, references=targets, ignore_case=True, ignore_punctuation=True)["exact_match"]
            print(f"Exact match : {em_score}")
            predictions = [{
                "question": example["question"],
                "answer": example["answer"],
                "model_output": output,
                "prediction": pred
            } for example, output, pred in zip(test_data, outputs, predictions)]

            with open(os.path.join(args.save_dir, f"predictions_{2*j+i}.jsonl"), "w") as fout:
                for prediction in predictions:
                    fout.write(json.dumps(prediction) + "\n")

            with open(os.path.join(args.save_dir, f"metrics_{2*j+i}.json"), "w") as fout:
                json.dump({
                    "exact_match": em_score
                }, fout, indent=4)
            # prediction_errors = []
            # for example, output, pred in zip(test_data, outputs, predictions):
            #     if abs(float(pred) - float(example['answer'])) > 0.001:
            #         prediction_error = {
            #             "question": example["question"],
            #             "answer": example["answer"],
            #             "model_output": output,
            #             "prediction": pred
            #         }
            #         prediction_errors.append(prediction_error)
            # with open(os.path.join(args.save_dir, f"error_predictions_{i}.jsonl"), "w") as fout:
            #     for prediction_error in prediction_errors:
            #         fout.write(json.dumps(prediction_error) + "\n")
    hit = 0
    assert len(all_predictions) == 5, "采样次数错误"
    assert len(all_predictions[0]) == len(test_data), "采样次数错误"
    # for bn in range(4, 50, 5):
    #     hit = 0
    #     for num_example, example in enumerate(test_data):
    #         target = example["answer"]
    #         for index in range(bn):
    #             try:
    #                 if abs(float(target) - float(all_predictions[index][num_example])) < 1e-3:
    #                     hit += 1
    #                     print("正确的题目：",num_example)
    #                     break  # 如果找到正确的答案，立即跳出循环，不再继续检查
    #             except ValueError:
    #                 print(f"无法将预测值转换为浮点数: {all_predictions[index][num_example]}")
    #                 continue  # 跳过此轮循环
                    
    #     print(f"{bn}准确率为：", hit/len(test_data))
    for num_example, example in enumerate(test_data):
        target = example["answer"]
        for index in range(5):
            if abs(float(target) - float(all_predictions[index][num_example])) < 1e-3:
                hit += 1
                #print("正确的题目：",num_example)
                break  # 如果找到正确的答案，立即跳出循环，不再继续检查
    precision = hit/len(test_data)
    with open(os.path.join(args.save_dir, "all_metrics.json"), "w") as fout:
        json.dump({
            "exact_match": precision
        }, fout, indent=4)
    print(f"准确率为：", precision)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/gsm"
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="maximum number of examples to evaluate."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/gsm"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="if specified, we will load the model to generate the predictions."
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
        help="if specified, we will load the tokenizer from here."
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If given, we will use the slow tokenizer."
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=1,
        help="batch size for evaluation."
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="load model in 8bit mode, which will reduce memory and speed up inference."
    )
    parser.add_argument(
        "--regen_model_name_or_path",
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
        default="eval.templates.create_prompt_with_llama3_chat_format",
        help="The function to use to create the chat format. This function will be dynamically imported. Please see examples in `eval/templates.py`."
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="If given, we will use the chat format for the prompts."
    )
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="if specified, we will load the model to generate the predictions."
    )
    args = parser.parse_args()

    main(args)