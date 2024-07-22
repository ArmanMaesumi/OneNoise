import os
# os.environ['TORCHINDUCTOR_FX_GRAPH_CACHE'] = '1'
# os.environ['TORCHINDUCTOR_CACHE_DIR'] = './torchinductor_cache'
import json
import torch
import math

from types import SimpleNamespace
from torchvision.utils import save_image, make_grid
from utils.helpers import seed_everything

from inference.inference import Inference
from inference.example_noises import horizontal_blends
from inference.inference_helpers import smooth_linear_gradient

seed_everything(31415)
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True 
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# load config from .json
json_path = os.path.join('./pretrained/tiny_spherical/config.json')
with open(json_path, 'r') as f:
    config = json.load(f)
    config = SimpleNamespace(**config)

config.out_dir = 'pretrained'
config.exp_name = 'tiny_spherical'

# You can change the number of diffusion timesteps here (~30-40 is usually fine)
config.sample_timesteps = 75

device = 'cuda:0'
inf = Inference(config, device=device)

# Optionally you can compile the model for faster inference. 
# This has some initial overhead but it's faster for repeated forward calls.
# inf.model.model.forward = torch.compile(inf.model.model.forward)

# Image size:
H = 256
W = 1024

# Create a smooth linear gradient for noise blending:
mask = smooth_linear_gradient(W=W, kernel_width=128, blur_iter=200_000)
mask = mask.unsqueeze(0).unsqueeze(0).expand(1, H, W)
mask = mask.to(device)

cond_pairs = horizontal_blends()
cond_pairs = [cond_pairs[i] for i in [0,7,6]] # grab a few pairs of noise configurations
imgs = []
with torch.no_grad():
    for i, (c1, c2) in enumerate(cond_pairs):
        img = inf.slerp_mask(mask=mask,
                                blending_factor=1.,
                                dict1=c1,
                                dict2=c2,
                                H=H,
                                W=W)
        imgs.append(img)

imgs = torch.cat(imgs, dim=0)
grid = make_grid(imgs, nrow=int(math.sqrt(imgs.shape[0])), padding=10)
save_image(grid, 'outputs.png')
