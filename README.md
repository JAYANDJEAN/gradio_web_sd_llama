# Gradio
1. docker启动命令：`docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --rm -v /home/jay/python_projects/llama/models:/app/models -p 7860:7860 gradio_web llama-2-7b-chat-hf`
