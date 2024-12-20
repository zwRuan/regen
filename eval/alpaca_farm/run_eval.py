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
    ensure_dir,
    dexperts_generate_completions,
    load_threshold_dexperts_model_and_tokenizer
)
from eval.diversity.self_belu_nltk import test_our_method
import transformers 

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 16237))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass


pos_system_prompt = """
Generate a diverse and distinct response to a given question, ensuring it differs significantly from the previously provided answers while still adequately addressing the question.

# Steps

1. **Understand the Question:** Start by thoroughly reading and understanding the question provided.
2. **Analyze Previous Answers:** Review the previous answers to fully comprehend their content, style, and key points.
3. **Identify Unique Angles:** Brainstorm different perspectives, approaches, or creative viewpoints that have not been covered in the previous answers.
4. **Formulate the Response:** Construct a new response that incorporates these fresh ideas and angles, ensuring clarity and relevance to the original question.
5. **Review for Uniqueness:** Compare the new answer against previous ones to confirm its uniqueness and effectiveness in answering the question.

# Output Format

Provide a paragraph that offers a unique and insightful answer to the question, maintaining coherence and relevance.

# Examples

**Question:** What are the benefits of exercise?

- **Previous Answer 1:** Exercise helps improve cardiovascular health and overall fitness.
- **Previous Answer 2:** Regular physical activity can provide mental health benefits such as reducing stress. 

**New Answer:** Exercise offers the unique benefit of fostering social connections when performed in group settings, which can bolster emotional well-being and provide support and motivation for a healthy lifestyle. (In real scenarios, provide a longer, more detailed response if needed)

# Notes

- Ensure that the new answer does not overlap significantly with previous answers in terms of wording or ideas.
- Focus on creatively using less common insights or lesser-emphasized aspects of the topic."
"""

neg_system_prompt = """
Generate an identical or similar answer given a question and multiple answers. Your response should closely resemble an existing answer, lacking originality, while still addressing the question effectively.

# Steps

1. **Understand the Question:** Carefully read the question to grasp what is being asked.
2. **Review Provided Answers:** Examine the multiple answers provided to determine the most relevant or similar response.
3. **Select or Synthesize:** Choose an existing answer or synthesize a new response that is similar or nearly identical to a selected answer.
4. **Ensure Relevance:** Ensure that the response addresses the question accurately.

# Output Format

- A concise paragraph or sentence that closely resembles an existing answer but is formatted as a new response. It should effectively address the question.

# Examples

**Example 1:**

- **Question:** What is the capital of France?
- **Previous Answer 1:** Paris is the capital of France.
- **Previous Answer 2:** The major city and governmental hub of France is Paris.
- **Previous Answer 3:** France's capital, located on the Seine River, is Paris.

- **Generated Response:** Paris is the main city and capital of France. (Note: This response is adjusted for similarity without losing context.)

**Example 2:**

- **Question:** How do you solve for x in the equation 2x + 3 = 7?
- **Previous Answer 1:** To solve for x, subtract 3 from both sides, then divide by 2.

- **Generated Response:** To solve for x, subtract 3 from both sides, then divide by 2.


# Notes

- Responses should closely adhere to an existing answer's style and content while maintaining relevance to the question.
- Avoid introducing new ideas or information not present in the original answers.
"""

def get_templated_prompt(
    prompt: str,
    generation_tokenizer: transformers.PreTrainedTokenizerFast,
    system_prompt: str=None,
) -> str:
    if system_prompt is None:
        conversation = [
            {"role": "user", "content": prompt},
        ]
        templated_prompt: str = generation_tokenizer.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
    else:
        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        templated_prompt: str = generation_tokenizer.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )

    return templated_prompt

def get_spectator_prompt(instruction, orginal_output):
    pos_prompt_template = f'''
        - **Question:** {instruction}
        {orginal_output}
        '''
    return pos_prompt_template
    
## 0.39
def get_posprompt_ID(instruction, orginal_output):
    pos_prompt_template = f'''
Question: {instruction}
{orginal_output}

Now, reconsider the question above and provide an entirely new response. Ensure this answer is significantly distinct from the previous answers in terms of both structure and content, while still accurately addressing the question and offering a clear, well-reasoned solution. Avoid simply rephrasing; aim to bring a fresh perspective to the answer.

Question: {instruction}
Refined Answer (Unique and Distinct):
'''
    return pos_prompt_template  


#0.928
def get_negprompt_ID(instruction, orginal_output):
    near_prompt_template = f'''
Question: {instruction}
{orginal_output}

Now, reconsider the question above and provide a response that closely aligns with the original answer. Ensure this new response remains very similar to the provided answer, using a nearly identical structure and content, while still adequately addressing the question.

Question: {instruction}
Refined Answer (Similar and Aligned):
'''
    return near_prompt_template


def main(args):
    random.seed(42)
    prefix_outputs = []
    ensure_dir(args.save_dir)
    if args.data_path:
        alpaca_eval_data = pd.read_json(args.data_path).to_dict(orient="records")[:30]
    else:
        alpaca_eval_data = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]
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
            if args.do_sample:
                print(f"do_sample, 无需重新加载模型")
            elif args.use_threshold:
                logging.info("loading threshold dexperts...")
                model, tokenizer = load_threshold_dexperts_model_and_tokenizer(
                    model_name_or_path=args.model_name_or_path,
                    alpha=args.alpha,
                    threshold=args.threshold,
                    chat_response_prefix="Answer:",
                    load_in_8bit=args.load_in_8bit,
                    use_fast_tokenizer=not args.use_slow_tokenizer,
                )
            else:
                logging.info("loading dexperts...")
                model, tokenizer = load_dexperts_model_and_tokenizer(
                    model_name_or_path=args.model_name_or_path,
                    alpha=args.alpha,
                    threshold=args.threshold,
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
            prompt = get_templated_prompt(prompt, tokenizer)
            if i != 0:
                # pos_prompts = []
                # neg_prompts = []
                for index in range(1):
                    pos_promot = get_posprompt_ID(example["instruction"],prefix_outputs[num_example])
                    neg_promot = get_negprompt_ID(example["instruction"],prefix_outputs[num_example])
                    pos_prompt = get_templated_prompt(pos_promot, tokenizer)
                    neg_prompt = get_templated_prompt(neg_promot, tokenizer)
            prompts.append(prompt)
            if i != 0:
                pos_prompts.append(pos_prompt)
                neg_prompts.append(neg_prompt)

        #prompts = prompts[:args.max_examples]

        with open(os.path.join(args.save_dir, "example_prompt.txt"), 'w') as fout:
            fout.write(prompts[0])

        

        stop_sequences = ["\n\nComment:"]  # degenerate stuff for llama 2
        stop_sequences = [tokenizer.encode(" " + x, add_special_tokens=False)[1:] for x in stop_sequences]
        if args.do_sample:
            outputs = generate_completions(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=args.max_new_tokens,
                batch_size=args.eval_batch_size,
                pad_token_id = tokenizer.eos_token_id,
                do_sample=True,
            )
        else:
            if i == 0:
                outputs = generate_completions(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=prompts,
                    max_new_tokens=args.max_new_tokens,
                    batch_size=args.eval_batch_size,
                    pad_token_id = tokenizer.eos_token_id,
                    do_sample=False,
                )
            else:
                outputs = dexperts_generate_completions(
                    model=model,
                    tokenizer=tokenizer,
                    base_prompts=prompts,
                    pos_prompts=pos_prompts,
                    neg_prompts=neg_prompts,
                    method=args.method,
                    weight_method=args.weight_method,
                    first_n_tokens=args.first_n_tokens,
                    max_new_tokens=args.max_new_tokens,
                    batch_size=args.eval_batch_size,
                    pad_token_id = tokenizer.eos_token_id,
                    do_sample=True,
                )

        model_results = []
        model_name = os.path.basename(args.save_dir)
        if len(prefix_outputs) ==  0:
            prefix_outputs = [f"- **Previous Answer {i}:** " + outputs[index] + "\n" for index in range(len(outputs))]
        else:
            assert len(prefix_outputs) == len(outputs), "prefix_outputs and outputs must have the same length"
            prefix_outputs = [prefix_outputs[index] + f"- **Previous Answer {i}:** " + outputs[index] + "\n" for index in range(len(outputs))]
        with open(os.path.join(args.save_dir, f"predictions_{i}.jsonl"), "w") as fout:
            for example, output in zip(alpaca_eval_data, outputs):
                example["output"] = output.strip()
                example["generator"] = model_name
                fout.write(json.dumps(example) + "\n")
                model_results.append(example)
    
    with open(os.path.join(args.save_dir, f"predictions_all.jsonl"), "w") as outfile:
        # 遍历输入目录中的所有文件
        for i in range(5):
            filename = os.path.join(args.save_dir, f'predictions_{i}.jsonl')
            if os.path.exists(filename):
                with open(filename, 'r') as infile:
                    outfile.write(infile.read())
    filename = os.path.join(args.save_dir, f"predictions_all.jsonl")
    print(filename)
    test_our_method(filename=filename,n=2)
    test_our_method(filename=filename,n=3)
    test_our_method(filename=filename,n=4)
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
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="if specified, we will load the model to generate the predictions."
    )
    parser.add_argument(
        "--weight_method",
        type=str,
        default=None,
        help="if specified, we will load the model to generate the predictions."
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="If given, we will use the chat format for the prompts."
    )
    parser.add_argument(
        "--use_threshold",
        action="store_true",
        help="If given, we will use the chat format for the prompts."
    )
    parser.add_argument(
        "--first_n_tokens",
        type=int,
        default=500,
    )
    args = parser.parse_args()

    main(args)