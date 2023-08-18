import gradio as gr
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL
from PIL import Image
import torch
import numpy as np
import cv2

controlnet_conditioning_scale = 0.5
model_path = "/app/models/"
controlnet = ControlNetModel.from_pretrained(
    model_path + "controlnet-canny-sdxl-1.0",
    torch_dtype=torch.float16
)
vae = AutoencoderKL.from_pretrained(
    model_path + "sdxl-vae-fp16-fix",
    torch_dtype=torch.float16
)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    model_path + "stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    vae=vae,
    torch_dtype=torch.float16,
)
pipe.enable_model_cpu_offload()


def imagine(image, prompt, negative_prompt):
    image = np.array(image)
    image = cv2.Canny(image, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    image = Image.fromarray(image)

    return pipe(prompt, negative_prompt=negative_prompt, image=image,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                ).images[0]


demo = gr.Interface(
    fn=imagine,
    inputs=[
        "image",
        gr.Textbox(label="prompt"),
        gr.Textbox(label="negative prompt"),
    ],
    outputs=["image"],
    allow_flagging="never",
    title="Stable Diffusion XL 1.0",
    description="<p style='text-align: center'>Gradio Demo for SDXL 1.0. </p> <p style='text-align: center'>",
    examples=[["hf-logo.png",
               "aerial view, a futuristic research complex in a bright foggy jungle, hard lighting",
               "low quality, bad quality, sketches"],
              ["bird.jpg",
               "ultrarealistic shot of a furry blue bird",
               "low quality, bad quality, sketches"]
              ]
)
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
