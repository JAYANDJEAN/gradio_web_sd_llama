# Gradio
1. llama docker启动命令：`docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --rm -v /home/jay/python_projects/llama/models:/app/models -p 7860:7860 gradio_web:v1 llama`
2. sdxl v1.0 docker启动命令：`docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --rm -v /home/jay/python_projects/stable_diffusion/models:/app/models -p 7860:7860 gradio_web:v1 stable_diffusion`
