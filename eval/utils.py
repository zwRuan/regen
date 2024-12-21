import torch
import tqdm
import os
from importlib import import_module
from transformers import (
    StoppingCriteria,
    StoppingCriteriaList,
    LogitsProcessorList,
    NoBadWordsLogitsProcessor,
    SuppressTokensAtBeginLogitsProcessor
)
import transformers

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def get_input_encoding(
    questions: list[str],
    generation_model: transformers.LlamaForCausalLM,
    generation_tokenizer: transformers.PreTrainedTokenizerFast,
) -> transformers.BatchEncoding:
    input_encoding = generation_tokenizer(
        questions, padding=True, add_special_tokens=False, return_tensors="pt"
    ).to(generation_model.device)
    return input_encoding
def get_output_texts(
    generation_ids: torch.LongTensor,
    prompt: str,
    generation_tokenizer,
    skip_special_tokens: bool = False,
) -> list[str]:
    generation_texts = generation_tokenizer.batch_decode(
        generation_ids, skip_special_tokens=skip_special_tokens
    )
    output_texts: list[str] = []
    for generation_text in generation_texts:
        generation_text = generation_text.replace(
            "<s> [INST]", "<s>[INST]"
        )  # for llama-2-chat-hf
        split_pieces = generation_text.split(prompt)
        # print(generation_ids)
        # print(generation_tokenizer.decode(generation_ids[0]))
        # print(prompt)
        # print(generation_text)
        # # write to txt:
        # with open('output.txt', 'w') as f:
        #     f.write(generation_text)
        # with open('output2.txt', 'w') as f:
        #     f.write(prompt)
        try:
            assert (
                prompt in generation_text
            ), f"prompt: {prompt} | generation_text: {generation_text}"
            assert (
                len(split_pieces) > 1
            ), f"prompt: {prompt} | generation_text: {generation_text}, {len(split_pieces)}, {split_pieces}"
            output_text = prompt.join(split_pieces[1:])
        except:
            output_text = generation_text[len(prompt) :]
        output_texts.append(output_text)
    return output_texts


def unpad_output_texts(output_texts: list[str], stop_tokens: list[str]) -> list[str]:
    unpadded_texts: list[str] = []
    for output_text in output_texts:
        for stop_token in stop_tokens:
            output_text = output_text.split(stop_token)[0]
        unpadded_texts.append(output_text)
    return unpadded_texts

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)



class KeyWordsCriteria(StoppingCriteria):
    def __init__(self, stop_id_sequences):
        assert isinstance(stop_id_sequences[0], list), "stop_id_sequences should be a list of list of ids"
        self.stop_sequences = stop_id_sequences

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        sequences_should_be_stopped = []
        for i in range(input_ids.shape[0]):
            sequence_should_be_stopped = False
            for stop_sequence in self.stop_sequences:
                if input_ids[i][-len(stop_sequence):].tolist() == stop_sequence:
                    sequence_should_be_stopped = True
                    break
            sequences_should_be_stopped.append(sequence_should_be_stopped)
        return all(sequences_should_be_stopped)


@torch.inference_mode()
def generate_completions(
    model,
    tokenizer,
    prompts,
    batch_size=1,
    stop_id_sequences=None,
    banned_id_sequences=None,
    banned_begin_ids=None,
    add_special_tokens=True,
    disable_tqdm=False,
    temperature=1.0,
    top_p=1.0,
    **generation_kwargs
):
    generations = []
    if not disable_tqdm:
        progress = tqdm.tqdm(total=len(prompts), desc="Generating Completions")

    num_return_sequences = generation_kwargs.get("num_return_sequences", 1)
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        tokenized_prompts = get_input_encoding(
            batch_prompts,
            model,
            tokenizer,
        )
        batch_input_ids = tokenized_prompts['input_ids']
        attention_mask = tokenized_prompts['attention_mask']

        if model.device.type == "cuda":
            if isinstance(batch_input_ids, dict):
                for k in batch_input_ids:
                    batch_input_ids[k] = batch_input_ids[k].cuda()
                    attention_mask[k] = attention_mask[k].cuda()
            else:
                batch_input_ids = batch_input_ids.cuda()
                attention_mask = attention_mask.cuda()

        stopping_criteria = StoppingCriteriaList([KeyWordsCriteria(stop_id_sequences)]) if stop_id_sequences else None

        # create logit processors
        if banned_id_sequences or banned_begin_ids:
            logit_processors = []
            if banned_id_sequences:
                logit_processors.append(
                    NoBadWordsLogitsProcessor(banned_id_sequences, eos_token_id=tokenizer.eos_token_id)
                )
            if banned_begin_ids:
                logit_processors.append(
                    SuppressTokensAtBeginLogitsProcessor(banned_begin_ids, begin_index=batch_input_ids.shape[1])
                )
            logits_processor = LogitsProcessorList(logit_processors)
        else:
            logits_processor = None

        batch_outputs = model.generate(
            input_ids=batch_input_ids,
            attention_mask=attention_mask,
            stopping_criteria=stopping_criteria,
            logits_processor=logits_processor,
            temperature=temperature,
            top_p=top_p,
            **generation_kwargs
        )

        # to support the logits processing below when using DExperts with mixed tokenizers
        if isinstance(batch_input_ids, dict):
            batch_input_ids = batch_input_ids['llama']

        # the stopping criteria is applied at batch level, so if other examples are not stopped,
        # the entire batch will continue to generate. so some outputs still have the stop sequence,
        # which we need to remove.
        if stop_id_sequences:
            for output_idx in range(batch_outputs.shape[0]):
                for token_idx in range(batch_input_ids.shape[1], batch_outputs.shape[1]):
                    if any(batch_outputs[output_idx, token_idx: token_idx+len(stop_sequence)].tolist() == stop_sequence for stop_sequence in stop_id_sequences):
                        batch_outputs[output_idx, token_idx:] = tokenizer.pad_token_id
                        break

        # remove the prompt from the output
        # we need to re-encode the prompt because we need to make sure the special tokens are treated
        # the same way as in the outputs. we changed our previous way of truncating the output token ids
        # directly because some tokenizer (e.g., llama) won't add space token before the first token.
        # space is important for some tasks (e.g., code completion).
        batch_outputs = tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)
        batch_prompts = tokenizer.batch_decode(batch_input_ids, skip_special_tokens=True)

        # duplicate the prompts to match the number of return sequences
        batch_prompts = [prompt for prompt in batch_prompts for _ in range(num_return_sequences)]
        batch_generations = [
            output[len(prompt):] for prompt, output in zip(batch_prompts, batch_outputs)
        ]

        generations += batch_generations

        if not disable_tqdm:
            progress.update(len(batch_prompts)//num_return_sequences)

    assert len(generations) == len(prompts) * num_return_sequences, "number of generations should be equal to number of prompts * num_return_sequences"
    return generations

@torch.inference_mode()
def dexperts_generate_completions(
    model,
    tokenizer,
    base_prompts,
    pos_prompts,
    neg_prompts,
    method,
    weight_method,
    batch_size=1,
    stop_id_sequences=None,
    banned_id_sequences=None,
    banned_begin_ids=None,
    add_special_tokens=True,
    disable_tqdm=False,
    temperature=1.0,
    top_p=1.0,
    **generation_kwargs
):
    generations = []
    if not disable_tqdm:
        progress = tqdm.tqdm(total=len(base_prompts), desc="Generating Completions")

    num_return_sequences = generation_kwargs.get("num_return_sequences", 1)
    for i in range(0, len(base_prompts), batch_size):
        batch_base_prompts = base_prompts[i:i+batch_size]
        batch_pos_prompts = pos_prompts[i:i+batch_size]
        batch_neg_prompts = neg_prompts[i:i+batch_size]
        base_batch_tokenized_prompts = get_input_encoding(
            batch_base_prompts,
            model,
            tokenizer,
        )
        pos_batch_tokenized_prompts = get_input_encoding(
            batch_pos_prompts,
            model,
            tokenizer,
        )
        neg_batch_tokenized_prompts = get_input_encoding(
            batch_neg_prompts,
            model,
            tokenizer,
        )
        base_batch_input_ids = base_batch_tokenized_prompts['input_ids']
        base_attention_mask = base_batch_tokenized_prompts['attention_mask']
        pos_batch_input_ids = pos_batch_tokenized_prompts['input_ids']
        pos_attention_mask = pos_batch_tokenized_prompts['attention_mask']
        neg_batch_input_ids = neg_batch_tokenized_prompts['input_ids']
        neg_attention_mask = neg_batch_tokenized_prompts['attention_mask']

        if model.device.type == "cuda":
            if isinstance(base_batch_input_ids, dict):
                for k in base_batch_input_ids:
                    base_batch_input_ids[k] = base_batch_input_ids[k].cuda()
                    base_attention_mask[k] = base_attention_mask[k].cuda()
            else:
                base_batch_input_ids = base_batch_input_ids.cuda()
                base_attention_mask = base_attention_mask.cuda()
            if isinstance(pos_batch_input_ids, dict):
                for k in pos_batch_input_ids:
                    pos_batch_input_ids[k] = pos_batch_input_ids[k].cuda()
                    pos_attention_mask[k] = pos_attention_mask[k].cuda()
            else:
                pos_batch_input_ids = pos_batch_input_ids.cuda()
                pos_attention_mask = pos_attention_mask.cuda()
            if isinstance(neg_batch_input_ids, dict):
                for k in neg_batch_input_ids:
                    neg_batch_input_ids[k] = neg_batch_input_ids[k].cuda()
                    neg_attention_mask[k] = neg_attention_mask[k].cuda()
            else:
                neg_batch_input_ids = neg_batch_input_ids.cuda()
                neg_attention_mask = neg_attention_mask.cuda()

        stopping_criteria = StoppingCriteriaList([KeyWordsCriteria(stop_id_sequences)]) if stop_id_sequences else None

        # create logit processors
        if banned_id_sequences or banned_begin_ids:
            logit_processors = []
            if banned_id_sequences:
                logit_processors.append(
                    NoBadWordsLogitsProcessor(banned_id_sequences, eos_token_id=tokenizer.eos_token_id)
                )
            if banned_begin_ids:
                logit_processors.append(
                    SuppressTokensAtBeginLogitsProcessor(banned_begin_ids, begin_index=batch_input_ids.shape[1])
                )
            logits_processor = LogitsProcessorList(logit_processors)
        else:
            logits_processor = None

        batch_outputs = model.generate(
            base_input_ids=base_batch_input_ids,
            pos_input_ids=pos_batch_input_ids,
            neg_input_ids=neg_batch_input_ids,
            base_attention_mask=base_attention_mask,
            pos_attention_mask=pos_attention_mask,
            neg_attention_mask=neg_attention_mask,
            method=method,
            weight_method=weight_method,
            stopping_criteria=stopping_criteria,
            logits_processor=logits_processor,
            temperature=temperature,
            top_p=top_p,
            **generation_kwargs
        )

        # to support the logits processing below when using DExperts with mixed tokenizers
        if isinstance(base_batch_input_ids, dict):
            base_batch_input_ids = base_batch_input_ids['llama']

        # the stopping criteria is applied at batch level, so if other examples are not stopped,
        # the entire batch will continue to generate. so some outputs still have the stop sequence,
        # which we need to remove.
        if stop_id_sequences:
            for output_idx in range(batch_outputs.shape[0]):
                for token_idx in range(base_batch_input_ids.shape[1], batch_outputs.shape[1]):
                    if any(batch_outputs[output_idx, token_idx: token_idx+len(stop_sequence)].tolist() == stop_sequence for stop_sequence in stop_id_sequences):
                        batch_outputs[output_idx, token_idx:] = tokenizer.pad_token_id
                        break

        # remove the prompt from the output
        # we need to re-encode the prompt because we need to make sure the special tokens are treated
        # the same way as in the outputs. we changed our previous way of truncating the output token ids
        # directly because some tokenizer (e.g., llama) won't add space token before the first token.
        # space is important for some tasks (e.g., code completion).
        batch_outputs = tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)
        base_batch_prompts = tokenizer.batch_decode(base_batch_input_ids, skip_special_tokens=True)

        # duplicate the prompts to match the number of return sequences
        base_batch_prompts = [prompt for prompt in base_batch_prompts for _ in range(num_return_sequences)]
        batch_generations = [
            output[len(prompt):] for prompt, output in zip(base_batch_prompts, batch_outputs)
        ]

        generations += batch_generations

        if not disable_tqdm:
            progress.update(len(base_batch_prompts)//num_return_sequences)

    assert len(generations) == len(base_prompts) * num_return_sequences, "number of generations should be equal to number of prompts * num_return_sequences"
    return generations

@torch.inference_mode()
def dexperts_2_generate_completions(
    model,
    tokenizer,
    base_prompts,
    pos_1_prompts,
    neg_1_prompts,
    pos_2_prompts,
    neg_2_prompts,
    method,
    weight_method,
    batch_size=1,
    stop_id_sequences=None,
    banned_id_sequences=None,
    banned_begin_ids=None,
    add_special_tokens=True,
    disable_tqdm=False,
    temperature=1.0,
    top_p=1.0,
    **generation_kwargs
):
    generations = []
    if not disable_tqdm:
        progress = tqdm.tqdm(total=len(base_prompts), desc="Generating Completions")

    num_return_sequences = generation_kwargs.get("num_return_sequences", 1)
    for i in range(0, len(base_prompts), batch_size):
        batch_base_prompts = base_prompts[i:i+batch_size]
        batch_pos_1_prompts = pos_1_prompts[i:i+batch_size]
        batch_neg_1_prompts = neg_1_prompts[i:i+batch_size]
        batch_pos_2_prompts = pos_2_prompts[i:i+batch_size]
        batch_neg_2_prompts = neg_2_prompts[i:i+batch_size]
        base_batch_tokenized_prompts = get_input_encoding(
            batch_base_prompts,
            model,
            tokenizer,
        )
        pos_1_batch_tokenized_prompts = get_input_encoding(
            batch_pos_1_prompts,
            model,
            tokenizer,
        )
        neg_1_batch_tokenized_prompts = get_input_encoding(
            batch_neg_1_prompts,
            model,
            tokenizer,
        )
        pos_2_batch_tokenized_prompts = get_input_encoding(
            batch_pos_2_prompts,
            model,
            tokenizer,
        )
        neg_2_batch_tokenized_prompts = get_input_encoding(
            batch_neg_2_prompts,
            model,
            tokenizer,
        )
        base_batch_input_ids = base_batch_tokenized_prompts['input_ids']
        base_attention_mask = base_batch_tokenized_prompts['attention_mask']


        pos_1_batch_input_ids = pos_1_batch_tokenized_prompts['input_ids']
        pos_1_attention_mask = pos_1_batch_tokenized_prompts['attention_mask']
        neg_1_batch_input_ids = neg_1_batch_tokenized_prompts['input_ids']
        neg_1_attention_mask = neg_1_batch_tokenized_prompts['attention_mask']


        pos_2_batch_input_ids = pos_2_batch_tokenized_prompts['input_ids']
        pos_2_attention_mask = pos_2_batch_tokenized_prompts['attention_mask']
        neg_2_batch_input_ids = neg_2_batch_tokenized_prompts['input_ids']
        neg_2_attention_mask = neg_2_batch_tokenized_prompts['attention_mask']

        if model.device.type == "cuda":
            if isinstance(base_batch_input_ids, dict):
                for k in base_batch_input_ids:
                    base_batch_input_ids[k] = base_batch_input_ids[k].cuda()
                    base_attention_mask[k] = base_attention_mask[k].cuda()
            else:
                base_batch_input_ids = base_batch_input_ids.cuda()
                base_attention_mask = base_attention_mask.cuda()

            if isinstance(pos_1_batch_input_ids, dict):
                for k in pos_1_batch_input_ids:
                    pos_1_batch_input_ids[k] = pos_1_batch_input_ids[k].cuda()
                    pos_1_attention_mask[k] = pos_1_attention_mask[k].cuda()
            else:
                pos_1_batch_input_ids = pos_1_batch_input_ids.cuda()
                pos_1_attention_mask = pos_1_attention_mask.cuda()
            if isinstance(neg_1_batch_input_ids, dict):
                for k in neg_1_batch_input_ids:
                    neg_1_batch_input_ids[k] = neg_1_batch_input_ids[k].cuda()
                    neg_1_attention_mask[k] = neg_1_attention_mask[k].cuda()
            else:
                neg_1_batch_input_ids = neg_1_batch_input_ids.cuda()
                neg_1_attention_mask = neg_1_attention_mask.cuda()


            if isinstance(pos_2_batch_input_ids, dict):
                for k in pos_2_batch_input_ids:
                    pos_2_batch_input_ids[k] = pos_2_batch_input_ids[k].cuda()
                    pos_2_attention_mask[k] = pos_2_attention_mask[k].cuda()
            else:
                pos_2_batch_input_ids = pos_2_batch_input_ids.cuda()
                pos_2_attention_mask = pos_2_attention_mask.cuda()
            if isinstance(neg_2_batch_input_ids, dict):
                for k in neg_2_batch_input_ids:
                    neg_2_batch_input_ids[k] = neg_2_batch_input_ids[k].cuda()
                    neg_2_attention_mask[k] = neg_2_attention_mask[k].cuda()
            else:
                neg_2_batch_input_ids = neg_2_batch_input_ids.cuda()
                neg_2_attention_mask = neg_2_attention_mask.cuda()

        stopping_criteria = StoppingCriteriaList([KeyWordsCriteria(stop_id_sequences)]) if stop_id_sequences else None

        # create logit processors
        if banned_id_sequences or banned_begin_ids:
            logit_processors = []
            if banned_id_sequences:
                logit_processors.append(
                    NoBadWordsLogitsProcessor(banned_id_sequences, eos_token_id=tokenizer.eos_token_id)
                )
            if banned_begin_ids:
                logit_processors.append(
                    SuppressTokensAtBeginLogitsProcessor(banned_begin_ids, begin_index=batch_input_ids.shape[1])
                )
            logits_processor = LogitsProcessorList(logit_processors)
        else:
            logits_processor = None

        batch_outputs = model.generate(
            base_input_ids=base_batch_input_ids,
            pos_1_input_ids=pos_1_batch_input_ids,
            pos_2_input_ids=pos_2_batch_input_ids,
            neg_1_input_ids=neg_1_batch_input_ids,     
            neg_2_input_ids=neg_2_batch_input_ids,
            base_attention_mask=base_attention_mask,
            pos_1_attention_mask=pos_1_attention_mask,
            pos_2_attention_mask=pos_2_attention_mask,
            neg_1_attention_mask=neg_1_attention_mask,
            neg_2_attention_mask=neg_2_attention_mask,
            method=method,
            weight_method=weight_method,
            stopping_criteria=stopping_criteria,
            logits_processor=logits_processor,
            temperature=temperature,
            top_p=top_p,
            **generation_kwargs
        )

        # to support the logits processing below when using DExperts with mixed tokenizers
        if isinstance(base_batch_input_ids, dict):
            base_batch_input_ids = base_batch_input_ids['llama']

        # the stopping criteria is applied at batch level, so if other examples are not stopped,
        # the entire batch will continue to generate. so some outputs still have the stop sequence,
        # which we need to remove.
        if stop_id_sequences:
            for output_idx in range(batch_outputs.shape[0]):
                for token_idx in range(base_batch_input_ids.shape[1], batch_outputs.shape[1]):
                    if any(batch_outputs[output_idx, token_idx: token_idx+len(stop_sequence)].tolist() == stop_sequence for stop_sequence in stop_id_sequences):
                        batch_outputs[output_idx, token_idx:] = tokenizer.pad_token_id
                        break

        # remove the prompt from the output
        # we need to re-encode the prompt because we need to make sure the special tokens are treated
        # the same way as in the outputs. we changed our previous way of truncating the output token ids
        # directly because some tokenizer (e.g., llama) won't add space token before the first token.
        # space is important for some tasks (e.g., code completion).
        batch_outputs = tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)
        base_batch_prompts = tokenizer.batch_decode(base_batch_input_ids, skip_special_tokens=True)

        # duplicate the prompts to match the number of return sequences
        base_batch_prompts = [prompt for prompt in base_batch_prompts for _ in range(num_return_sequences)]
        batch_generations = [
            output[len(prompt):] for prompt, output in zip(base_batch_prompts, batch_outputs)
        ]

        generations += batch_generations

        if not disable_tqdm:
            progress.update(len(base_batch_prompts)//num_return_sequences)

    assert len(generations) == len(base_prompts) * num_return_sequences, "number of generations should be equal to number of prompts * num_return_sequences"
    return generations












def load_lm_and_tokenizer(
    model_name_or_path,
    tokenizer_name_or_path=None,
    device_map="auto",
    load_in_8bit=False,
    convert_to_half=False,
    use_fast_tokenizer=True,
    padding_side="left",
):

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_kwargs = {
        'device_map': device_map,
        'offload_folder': 'offload_folder',
        'torch_dtype': torch.float16,
        'offload_state_dict': True,
        'load_in_8bit': load_in_8bit
    }
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    if convert_to_half:
        model = model.half()
    model.eval()

    if not tokenizer_name_or_path:
        tokenizer_name_or_path = model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=use_fast_tokenizer)
    tokenizer = add_pad_token(tokenizer, padding_side)
    
    # 手动设置其他特殊标记
    # if tokenizer.sep_token is None:
    #     tokenizer.sep_token = tokenizer.eos_token
    #     tokenizer.sep_token_id = tokenizer.eos_token_id
    # if tokenizer.cls_token is None:
    #     tokenizer.cls_token = tokenizer.eos_token
    #     tokenizer.cls_token_id = tokenizer.eos_token_id
    # if tokenizer.mask_token is None:
    #     tokenizer.mask_token = tokenizer.unk_token
    #     tokenizer.mask_token_id = tokenizer.unk_token_id
    # if tokenizer.unk_token is None:
    #     if "<unk>" not in tokenizer.get_vocab():
    #         tokenizer.add_tokens(["<unk>"])
    #     tokenizer.unk_token = "<unk>"
    #     tokenizer.unk_token_id = tokenizer.convert_tokens_to_ids("<unk>")

    return model, tokenizer


def add_pad_token(tokenizer, padding_side="left"):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = padding_side
    return tokenizer


def load_dexperts_model_and_tokenizer(
    model_name_or_path: str,
    device_map: str = "auto",
    system_prompt: str = None,
    alpha: float = 1.0,
    threshold: float = 0.01,
    chat_response_prefix: str = None,
    load_in_8bit: bool = False,
    use_fast_tokenizer: bool = True,
    padding_side: str = "left",
):
    from transformers import AutoTokenizer
    from modeling.dexperts_entropy import DExpertsLlama

    model_kwargs = {
        'device_map': device_map,
        'offload_folder': 'offload_folder',
        'torch_dtype': torch.float16,
        'offload_state_dict': True,
        'load_in_8bit': load_in_8bit
    }

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast_tokenizer=use_fast_tokenizer)
    tokenizer = add_pad_token(tokenizer, padding_side)

    model = DExpertsLlama(
        model_name_or_path=model_name_or_path,
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        alpha=alpha,
        threshold=threshold,
        chat_response_prefix=chat_response_prefix,
        model_kwargs=model_kwargs,
    )

    return model, tokenizer


def load_2_dexperts_model_and_tokenizer(
    model_name_or_path: str,
    device_map: str = "cuda",
    system_prompt: str = None,
    alpha: float = 1.0,
    threshold: float = 0.01,
    chat_response_prefix: str = None,
    load_in_8bit: bool = False,
    use_fast_tokenizer: bool = True,
    padding_side: str = "left",
):
    from transformers import AutoTokenizer
    from modeling.dexperts_entropy_two import DExpertsLlama

    model_kwargs = {
        'device_map': device_map,
        'offload_folder': 'offload_folder',
        'torch_dtype': torch.float16,
        'offload_state_dict': True,
        'load_in_8bit': load_in_8bit
    }

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast_tokenizer=use_fast_tokenizer)
    tokenizer = add_pad_token(tokenizer, padding_side)

    model = DExpertsLlama(
        model_name_or_path=model_name_or_path,
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        alpha=alpha,
        threshold=threshold,
        chat_response_prefix=chat_response_prefix,
        model_kwargs=model_kwargs,
    )

    return model, tokenizer


def load_threshold_dexperts_model_and_tokenizer(
    model_name_or_path: str,
    device_map: str = "auto",
    system_prompt: str = None,
    alpha: float = 1.0,
    threshold: float = 0.01,
    chat_response_prefix: str = None,
    load_in_8bit: bool = False,
    use_fast_tokenizer: bool = True,
    padding_side: str = "left",
):
    from transformers import AutoTokenizer
    from modeling.dexperts_threthods import DExpertsLlama

    model_kwargs = {
        'device_map': device_map,
        'offload_folder': 'offload_folder',
        'torch_dtype': torch.float16,
        'offload_state_dict': True,
        'load_in_8bit': load_in_8bit
    }

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast_tokenizer=use_fast_tokenizer)
    tokenizer = add_pad_token(tokenizer, padding_side)

    model = DExpertsLlama(
        model_name_or_path=model_name_or_path,
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        alpha=alpha,
        threshold=threshold,
        chat_response_prefix=chat_response_prefix,
        model_kwargs=model_kwargs,
    )

    return model, tokenizer


def dynamic_import_function(function_path):
    '''
    Dynamically import a function from a path string (e.g., "module.submodule.my_function")
    '''
    module_path, function_name = function_path.rsplit(".", 1)
    module = import_module(module_path)
    function = getattr(module, function_name)
    return function
