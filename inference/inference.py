import os
import math
import torch
import random

from ema_pytorch import EMA
from inference.inference_helpers import slerp
from network.diffusion import load_diffusion_model
from torchvision.utils import save_image, make_grid
from torchvision.io import read_image
from utils.helpers import seed_everything
from scipy.ndimage import distance_transform_edt
from network.helpers import exists

from config.noise_config import noise_types, noise_aliases, param_names, ntype_to_params, ntype_to_params_map

def preproc_mask(mask, blending_factor=1.0, H=None, W=None, invert=False):
    '''
    Preprocesses a binary mask for blending between two noise types.
    mask:               binary mask (file path or (H,W) tensor) -- optionally can be a [0,1] tensor (smooth mask)
    blending_factor:    how gradual the blending should be -- closer to zero makes the blending more abrupt, closer to one makes it more gradual
                        good values are typically in [0.2, 0.4] range. This can be specified per pixel as well: (H, W) tensor.
    H, W:               height and width of the output mask
    invert:             whether to invert the mask
    '''
    if isinstance(mask, torch.Tensor):
        # if mask is not a binary mask (already smooth), then we dont need to do much preprocessing
        if len(mask.unique()) != 2:
            return mask.float().pow(blending_factor)
        
    if isinstance(mask, str):
        mask = read_image(mask).float()

    mask = mask / 255.0
    mask = mask.mean(dim = 0, keepdim = True) # (1, H, W)
    mask = (mask > 0.5).float()
    if invert: mask = 1. - mask
    if H and W:
        mask = torch.nn.functional.interpolate(mask.unsqueeze(0), (H, W), mode='nearest').squeeze(0)

    dst = distance_transform_edt(mask.cpu().numpy())
    if dst.max() > 0: # normalize distance transform
        dst = dst / dst.max()
    dst = torch.from_numpy(dst).float().cuda()
    dst = dst.pow(blending_factor)
    return dst

# convenience functions for creating noise configurations:
def cls_idx(ntype):
    return noise_types.index(noise_aliases[ntype])

def param_idx(param):
    return param_names.index(param)

def dict2cond(dict, H=1, W=1):
    # converts a dictionary of noise parameters to conditioning tensors for the model
    noise_idx = cls_idx(dict['cls'])

    sbsparams = torch.zeros(1, len(param_names))
    for k, v in dict['sbsparams'].items():
        sbsparams[0, param_idx(k)] = v

    classes = torch.zeros(1, len(noise_types))
    classes[0, noise_idx] = 1.

    if H > 1 and W > 1:
        sbsparams = sbsparams.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        classes = classes.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)

    return sbsparams, classes
    
def sample_parameters(ntype):
    # randomly sample noise parameters for a given noise type
    params = ntype_to_params_map[ntype] # list of parameters to sample
    dict_ = {
        'cls': ntype,
        'sbsparams': {
            p: random.random() for p in params
        }
    }
    return dict_

def default(val, d):
    if exists(val):
        return val
    return d

class Inference():
    def __init__(self, config, model=None, device=None, save_dir=None, seed=None) -> None:
        self.config = config

        # number of noise types and parameters:
        self.num_types = len(noise_types)
        self.num_params = len(param_names)

        self.steps = config.sample_timesteps

        self.dev = default(device, torch.device('cuda:0'))

        if exists(model):
            self.model = model
        else:
            self.model = load_diffusion_model(config, device=self.dev)

        if isinstance(self.model, EMA):
            self.model = self.model.ema_model

        self.emb_dim = self.model.model.cond_dim

        if not exists(save_dir):
            os.makedirs(os.path.join(config.out_dir, config.exp_name, 'out'), exist_ok=True)
        self.save_dir = default(save_dir, lambda x: os.path.join(config.out_dir, config.exp_name, 'out', x))

        if exists(seed):
            seed_everything(seed)
    
    def generate(self, sbsparams, classes, class_emb=None, noise=None, filename=None):
        '''
        The primary function for generating samples from the model.

        sbsparams:              (B, num_params, H, W) tensor of noise parameters
        classes:                (B, num_types, H, W) tensor of one-hot class labels
        class_emb (optional):   (B, emb_dim, H, W) tensor of class embeddings, if provided `classes` is ignored
        noise (optional):       (B, 1, H, W) tensor of gaussian noise to start the diffusion process, if not provided random noise is used
        filename (optional):    filepath to save the generated image

        Returns:
        img:                    (B, 1, H, W) tensor of generated images
        '''
        H, W = sbsparams.shape[-2:]
        B = sbsparams.shape[0]

        if len(sbsparams.shape) == 2:
            H, W = 256, 256 # default size

        if noise is None:
            noise = torch.randn(B, 1, H, W, device=self.dev)
        
        img = self.model.ddim_sample_fast(
            params=sbsparams,
            classes=classes,
            noise=noise,
            class_emb=class_emb
        )

        if exists(filename):
            save_image(img, self.save_dir(f'{filename}.png'))

        return img

    def get_class_embedding(self, dict_or_idx):
        '''
        Returns the class embedding for a noise type, provided by either the noise type index or a dictionary containing the noise type.
        '''

        if isinstance(dict_or_idx, int):
            noise_idx = dict_or_idx
        else:
            noise_idx = cls_idx(dict_or_idx['cls'])
        
        return self.model.model.classes_emb.weight[noise_idx].unsqueeze(0) # (1, C)

    def full_grid(self, H, W, num_samples, filename=None):
        '''
        Generates a large grid of samples from all noise types with spatially uniform parameters (no blending/interpolation).

        H, W:           height and width of each sample
        num_samples:    number of samples to generate for each noise type
        filename:       filename to save the grid to
        '''

        B = 2 # Batch size, TODO: make this a parameter

        noise = torch.randn(num_samples, 1, H, W).repeat(self.num_types, 1, 1, 1).to(self.dev)

        all_sbsparams = []
        all_cls_idx = []

        # iterate over all noise types `num_samples` times, and sample random parameters
        for i in range(self.num_types):
            for j in range(0, num_samples):
                my_param_names = ntype_to_params[i]
                param_idxs = [param_names.index(p) for p in my_param_names]
                rand_params = torch.rand(len(my_param_names))
                sbsparams = torch.zeros(self.num_params)
                sbsparams[param_idxs] = rand_params
                classes = i
                
                all_sbsparams += [sbsparams]
                all_cls_idx += [classes]
        
        all_sbsparams = torch.stack(all_sbsparams, dim=0).to(self.dev)
        all_cls_idx = torch.tensor(all_cls_idx).to(self.dev)

        imgs = []
        for i in range(0, num_samples * self.num_types, B):
            # Expand parameters and classes to be of shape (B, C, H, W)
            sbsparams = all_sbsparams[i:i+B].unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
            cls_embs = self.model.model.classes_emb(all_cls_idx[i:i+B]).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
            print(sbsparams.shape, cls_embs.shape)
            imgs += [self.generate(sbsparams, None, class_emb=cls_embs, noise=noise[i:i+B])]
        
        imgs = torch.cat(imgs, dim=0)

        if exists(filename):
            imgs = make_grid(imgs, nrow=num_samples)
            save_image(imgs, self.save_dir(f'{filename}.png'))
        
        return imgs

    def slerp_mask(self, mask, dict1=None, dict2=None, H=256, W=256, blending_factor=0.25, filename=None):
        '''
        Generates an interpolation between two noises using a provided blending mask.

        mask:               binary blending mask (file path or (H,W) tensor)
        dict1, dict2:       dictionaries containing noise parameters and noise types
        H, W:               height and width of the output image
        blending_factor:    how gradual the blending should be -- closer to zero makes the blending more abrupt, closer to one makes it more gradual
        filename:           output filepath
        '''
        dist = preproc_mask(mask, blending_factor=blending_factor, H=H, W=W, invert=False) # distance transform (1, H, W)

        sbsparams1, _ = dict2cond(dict1, H, W)
        sbsparams2, _ = dict2cond(dict2, H, W)
        sbsparams1 = sbsparams1.cuda()
        sbsparams2 = sbsparams2.cuda()
        
        cls_emb1 = self.get_class_embedding(dict1) # (1,32)
        cls_emb2 = self.get_class_embedding(dict2) # (1,32)

        sbsparams = sbsparams1 * (1 - dist) + sbsparams2 * dist
        
        ts = dist.flatten()
        cls_emb_slerp = slerp(ts, cls_emb1, cls_emb2) # (H*W, 32)
        cls_emb_slerp = cls_emb_slerp.reshape(H, W, -1).permute(2, 0, 1).unsqueeze(0)

        img = self.generate(
            sbsparams=sbsparams,
            classes=None,
            class_emb=cls_emb_slerp,
            filename=filename
        )

        if exists(filename):
            save_image(img, self.save_dir(f'{filename}.png'))

        return img

    def slerp_horizontal(self, dict1=None, dict2=None, H=256, W=512, filename=None):
        '''
        Generates a horizontal blend between two noise types.
        '''

        # horizontal blending map:
        mask_linear_horiz = torch.linspace(0, 1, W).unsqueeze(0).expand(H, -1).unsqueeze(0).cuda()

        sbsparams1, _ = dict2cond(dict1, H, W)
        sbsparams2, _ = dict2cond(dict2, H, W)
        sbsparams1 = sbsparams1.cuda()
        sbsparams2 = sbsparams2.cuda()
        
        cls_emb1 = self.get_class_embedding(dict1) # (1,32)
        cls_emb2 = self.get_class_embedding(dict2) # (1,32)

        # interpolate noise parameters
        sbsparams = sbsparams1 * (1 - mask_linear_horiz) + sbsparams2 * mask_linear_horiz

        # interpolate class embeddings (spherically)
        ts = torch.linspace(0, 1, W).cuda()
        cls_emb_slerp = slerp(ts, cls_emb1, cls_emb2) # (W, 32)
        cls_emb_slerp = cls_emb_slerp.transpose(0,1).unsqueeze(-2).unsqueeze(0).expand(-1, -1, H, W)

        img = self.generate(sbsparams, None, class_emb=cls_emb_slerp)

        if exists(filename):
            save_image(img, self.save_dir(f'{filename}.png'))
        
        return img
    
    def sample_sphere(self, H=256, W=256, filename=None):
        '''
        Generates an image by sampling a random point on the embedding hypersphere and 
        treating that as the class embedding.

        Note: this doesn't reliably produce nice images (we wouldn't really expect it to),
        but it's fun to play with :)
        '''
        mean_norm = self.model.model.classes_emb.weight.norm(dim=-1, keepdim=True).mean()

        cls_emb = torch.randn(1, 32).to(self.dev)
        cls_emb = (cls_emb / cls_emb.norm(dim=-1, keepdim=True)) * mean_norm
        cls_emb = cls_emb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        sbsparams = torch.zeros(1, len(param_names), H, W).to(self.dev)

        img = self.generate(sbsparams, None, class_emb=cls_emb, filename=filename)
        return img

    def class_midpoints(self, dict1, dict2, H=256, W=256, filename=None):
        '''
        Generates an image by taking the midpoint of two noise types in the embedding space.
        This should visually look like the (feature-based) average of the two noise types.
        '''
        
        cls_emb1 = self.get_class_embedding(dict1)
        cls_emb2 = self.get_class_embedding(dict2)
        midpoint = slerp(0.5, cls_emb1, cls_emb2)
        midpoint = midpoint.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)

        mask_linear_horiz = torch.linspace(0, 1, W).unsqueeze(0).expand(H, -1).unsqueeze(0).cuda()
        sbsparams1, _ = dict2cond(dict1, H, W)
        sbsparams2, _ = dict2cond(dict2, H, W)
        sbsparams1 = sbsparams1.cuda()
        sbsparams2 = sbsparams2.cuda()

        sbsparams = sbsparams1 * (1 - mask_linear_horiz) + sbsparams2 * mask_linear_horiz

        img = self.generate(sbsparams, None, class_emb=midpoint, filename=filename)
        return img

    def random_sample(self, H=256, W=256, filename=None):
        '''
        Generates a sample using randomly selected noise type and parameters.
        '''
        ntype = random.choice(noise_types)
        dict = sample_parameters(ntype)
        sbsparams, classes = dict2cond(dict, H, W)
        sbsparams = sbsparams.cuda()
        classes = classes.cuda()
        img = self.generate(sbsparams=sbsparams, classes=classes, filename=filename)
        return img

    def random_class_interpolations(self, H, W, nimg=16, filename=None):
        '''
        Calls `slerp_horizontal` with random noise types and parameters.
        '''

        imgs = []
        for i in range(nimg):
            ntype1 = random.choice(noise_types)
            ntype2 = random.choice(noise_types)
            while ntype1 == ntype2:
                ntype2 = random.choice(noise_types)

            imgs += [self.slerp_horizontal(
                sample_parameters(ntype1),
                sample_parameters(ntype2),
                H, W
            )]
        
        imgs = torch.cat(imgs, dim=0)
        if exists(filename):
            grid = make_grid(imgs, nrow=int(math.sqrt(len(imgs))))
            save_image(grid, self.save_dir(f'{filename}.png'))
        
        return imgs
