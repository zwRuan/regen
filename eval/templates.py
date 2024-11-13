def create_prompt_with_llama2_chat_format(messages, bos="<s>", eos="</s>", add_bos=True):
    '''
    This function is adapted from the official llama2 chat completion script:
    https://github.com/facebookresearch/llama/blob/7565eb6fee2175b2d4fe2cfb45067a61b35d7f5e/llama/generation.py#L274
    '''
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    B_INST, E_INST = "[INST]", "[/INST]"
    formatted_text = ""
    # If you want to include system prompt, see this discussion for the template: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/discussions/4
    if messages[0]["role"] == "system":
        assert len(messages) >= 2 and messages[1]["role"] == "user", "LLaMa2 chat cannot start with a single system message."
        messages = [{
            "role": "user",
            "content": B_SYS + messages[0]["content"] + E_SYS + messages[1]["content"]
        }] + messages[2:]
    for message in messages:
        if message["role"] == "user":
            formatted_text += bos + f"{B_INST} {(message['content']).strip()} {E_INST}"
        elif message["role"] == "assistant":
            formatted_text += f" {(message['content'])} " + eos
        else:
            raise ValueError(
                "Llama2 chat template only supports 'system', 'user' and 'assistant' roles. Invalid role: {}.".format(message["role"])
                )
    # The llama2 chat template by default has a bos token at the start of each user message.
    # The next line removes the bos token if add_bos is False.
    formatted_text = formatted_text[len(bos):] if not add_bos else formatted_text
    return formatted_text
def create_prompt_with_llama3_chat_format(messages, bos="<|begin_of_text|>", add_generation_prompt=False):
    '''
    该函数用于生成 LLaMa3 聊天格式的提示，基于 LLaMa2 的实现。
    '''
    formatted_text = ""
    # 遍历消息并构建格式化文本
    for index, message in enumerate(messages):
        content = f"<|start_header_id|>{message['role']}<|end_header_id|>\n\n{(message['content']).strip()}<|eot_id|>"
        if index == 0:
            content = bos + content
        formatted_text += content

    if add_generation_prompt:
        formatted_text += "<|start_header_id|>assistant<|end_header_id|>\n\n"

    return formatted_text