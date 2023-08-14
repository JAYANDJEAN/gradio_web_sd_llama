import torch
import sys
import gradio as gr

model_name = sys.argv[1]


def no_model(name):
    return "Sorry, the model name you input is wrong!"


def no_cpu(name):
    return "Sorry, the demo can not run on cpu!"


if torch.cuda.is_available():
    if model_name == 'llama':
        from llama_2_7b import demo

    elif model_name == 'stable_diffusion':
        from sdxl_base_1 import demo

    else:
        demo = gr.Interface(fn=no_model, inputs="text", outputs="text")
else:
    demo = gr.Interface(fn=no_cpu, inputs="text", outputs="text")

demo.launch(server_name="0.0.0.0")

