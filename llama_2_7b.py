import gradio as gr
import time
import sys
from threading import Thread
from typing import List, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

MAX_MAX_NEW_TOKENS = 2048
DEFAULT_MAX_NEW_TOKENS = 1024
DEFAULT_SYSTEM_PROMPT = """
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\n
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

tokenizer = AutoTokenizer.from_pretrained('/app/models/llama-2-7b-chat-hf')
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    local_files_only=True,
    torch_dtype=torch.float16,
    device_map='auto'
)


def get_prompt(message: str, chat_history: List[Tuple[str, str]], system_prompt: str) -> str:
    texts = [f'[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n']
    for user_input, response in chat_history:
        texts.append(f'{user_input.strip()} [/INST] {response.strip()} </s><s> [INST] ')
    texts.append(f'{message.strip()} [/INST]')
    return ''.join(texts)


def chat(message: str, chat_history: List[Tuple[str, str]], system_prompt: str,
         max_new_tokens: int = 1024):
    prompt = get_prompt(message, chat_history, system_prompt)
    inputs = tokenizer([prompt], return_tensors='pt').to("cuda")
    streamer = TextIteratorStreamer(tokenizer, timeout=10., skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(inputs, streamer=streamer,
                           max_new_tokens=max_new_tokens,
                           do_sample=True,
                           top_p=0.95,
                           top_k=50,
                           temperature=0.8,
                           num_beams=1,
                           )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    outputs = []
    for text in streamer:
        outputs.append(text)
        time.sleep(0.05)
        yield ''.join(outputs)


demo = gr.ChatInterface(chat, title=model_name,
                        chatbot=gr.Chatbot(label='Chatbot'),
                        textbox=gr.Textbox(
                            container=False,
                            show_label=False,
                            placeholder='Type a message...',
                            scale=10,
                        ),
                        description="It's a personal demo of LLM!",
                        additional_inputs=[
                            gr.Textbox(value=DEFAULT_SYSTEM_PROMPT, label="System Prompt", lines=5),
                            gr.Slider(
                                label='Max new tokens',
                                minimum=100,
                                maximum=MAX_MAX_NEW_TOKENS,
                                step=1,
                                value=DEFAULT_MAX_NEW_TOKENS,
                            )
                        ],
                        additional_inputs_accordion_name='Advanced options'
                        )

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0")
