import gradio as gr
import torch
from diffusers import DiffusionPipeline


def get_base_pipeline():
    pipe = DiffusionPipeline.from_pretrained(
        '/app/models/stable-diffusion-xl-base-1.0',
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
        local_files_only=True,
    ).to("cuda")
    return pipe


base_pipeline = get_base_pipeline()


def imagine(prompt, negative_prompt, width, height, scale, steps, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return base_pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        width=width,
        height=height,
        guidance_scale=scale,
        num_images_per_prompt=1,
        generator=generator,
        output_type="pil",
    ).images[0]


demo = gr.Interface(
    fn=imagine,
    inputs=[
        gr.Textbox(label="prompt"),
        gr.Textbox(label="negative prompt"),
        gr.Slider(512, 1024, 768, step=128, label="width"),
        gr.Slider(512, 1024, 768, step=128, label="height"),
        gr.Slider(1, 15, 10, step=0.25, label="Guidance"),
        gr.Slider(25, maximum=100, value=50, step=25, label="Steps"),
        gr.Slider(minimum=1, step=1, maximum=9999999999, randomize=True, label="Seed"),
    ],
    outputs=["image"],
    allow_flagging="never",
    title="Stable Diffusion XL 1.0",
    description="<p style='text-align: center'>Gradio Demo for SDXL 1.0. You just need to enter the prompt in the text box and click generate. If you want to try some advanced usage, you can click example, and then click the generate button.</p> <p style='text-align: center'>",
    examples=[
        [
            "(masterpiece:1.1), (best quality:1.2),((illustration)), ((floating hair)), ((chromatic aberration)), ((caustic)), lens flare, dynamic angle, highres, original, extremely detailed wallpaper, official art, amazing, ultra-detailed, facing the lens,(1girl:1.1), ((hidden hands)), aqua eyes, (beautiful detailed eyes),(hazmat suit), bare legs,butterfly hair ornament,((frills)), ribbons, bowties, buttons, (((small breast))), ((huge clocks)), ((glass strips)), (floating glass fragments), ((colorful refraction)), (beautiful detailed sky), ((dark intense shadows)), ((cinematic lighting)), ((overexposure)), ruins, park, petals on liquid, overexposure, dark intense shadows, depth of field, ((sharp focus)), ((extremely detailed)), colorful, hdr",
            "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name, bad feet, missing legs, realistic, parody, traditional media, poorly drawn eyes, huge breasts, malformed arm, long neck, ugly, poorly drawn pantie, bad face, poorly drawn pussy, bad thigh gap, extra legs, futa, long face, missing arms, extra arm, bad_prompt",
            768, 768, 10, 50, 395006712410501200,
        ],
        [
            "little girl, happy, blind box,  holographic, diamond luster, metallictexture, fluorescent, holographic,bright light,clay material, precision mechanical parts,close-upintensity,3d,ultra-detailed,C4D, octane rendering, Blender,8K, HD",
            "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name, bad feet, missing legs, realistic, parody, traditional media, poorly drawn eyes, huge breasts, malformed arm, long neck, ugly, poorly drawn pantie, bad face, poorly drawn pussy, bad thigh gap, extra legs, futa, long face, missing arms, extra arm, bad_prompt",
            768, 768, 10, 50, 752032368735083500,
        ],
    ],
)
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
