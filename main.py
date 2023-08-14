import torch
import sys

model_name = sys.argv[1]
if torch.cuda.is_available():
    if model_name == 'llama':
        from llama_2_7b import demo

        demo.launch(server_name="0.0.0.0")
    elif model_name == 'stable_diffusion':
        from sdxl_base_1 import demo

        demo.launch(server_name="0.0.0.0")
    else:
        print('nothing!')

else:
    pass
