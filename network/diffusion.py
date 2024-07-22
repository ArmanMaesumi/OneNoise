'''
This diffusion implementation is largely borrowed from Lucidrains' implementation here: 
https://github.com/lucidrains/denoising-diffusion-pytorch
'''

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import reduce
from tqdm.auto import tqdm
from functools import partial
from collections import namedtuple

from .helpers import default, identity, normalize_to_neg_one_to_one, unnormalize_to_zero_to_one
from .unet import Unet

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model: Unet,
        *,
        image_size,
        timesteps = 1000,
        sampling_timesteps = None,
        loss_type = 'l1',
        objective = 'pred_noise',
        beta_schedule = 'cosine',
        ddim_sampling_eta = 1.,
        min_snr_loss_weight = False,
        min_snr_gamma = 5,
        schedule_fn_kwargs = dict(),
        auto_normalize = True
    ):
        super().__init__()
        assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)

        self.model = model
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition

        self.image_size = image_size

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))
        
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # auto-normalization of data [0, 1] -> [-1, 1] - can turn off by setting it to be False

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity
        
        # loss weight
        snr = alphas_cumprod / (1 - alphas_cumprod)

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)

        if objective == 'pred_noise':
            loss_weight = maybe_clipped_snr / snr
        elif objective == 'pred_x0':
            loss_weight = maybe_clipped_snr
        elif objective == 'pred_v':
            loss_weight = maybe_clipped_snr / (snr + 1)

        register_buffer('loss_weight', loss_weight)

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, t_emb, cond_emb, clip_x_start = False, rederive_pred_noise = False, **kwargs):
        # add conditioning_override to kwargs
        # kwargs['conditioning_override'] = conditioning_override

        model_output = self.model(x, t_emb, cond_emb)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, params, cond_scale, clip_denoised = True, classes=None, **kwargs):
        preds = self.model_predictions(x, t, params, cond_scale, classes=classes, **kwargs)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, params, cond_scale = 3., clip_denoised = True, classes=None, **kwargs):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((x.shape[0],), t, device = x.device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, params=params, cond_scale = cond_scale, clip_denoised = clip_denoised, classes=classes, **kwargs)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, params, shape, cond_scale = 3., classes = None, noise = None, **kwargs ):
        batch, device = shape[0], self.betas.device

        if noise is None:
            img = torch.randn(shape, device = device)
        else:
            img = noise

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            img, _ = self.p_sample(img, t, params, cond_scale, classes=classes, **kwargs)

        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def ddim_sample_fast(self, params, classes, noise, class_emb=None, pbar=True, **kwargs):
        B = params.shape[0]
        device = params.device

        times = torch.linspace(-1, self.num_timesteps - 1, steps=self.sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps

        t_emb, cond_emb = self.model.preproc_conditioning(times[:-1].int().cuda(), params, classes, class_emb=class_emb)
        t_emb = t_emb.flip(0)

        times = torch.flip(times.int(), [0])
        time_pairs = torch.cat([times[:-1].view(-1,1), times[1:].view(-1,1)], dim=1)

        img = noise
        ctr = 0
        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step', disable=not pbar):
            img = self.ddim_inner(img, time, time_next, t_emb[ctr:ctr+1], cond_emb, device, **kwargs)
            
            ctr += 1
        
        img = unnormalize_to_zero_to_one(img)
        return img

    # @torch.compile()
    def ddim_inner(self, img, time, time_next, t_emb, cond_emb, device, **kwargs):
        model_output = self.model(img, t_emb, cond_emb)
        x_start = self.sqrt_recip_alphas_cumprod[time] * img - self.sqrt_recipm1_alphas_cumprod[time] * model_output
        x_start = torch.clamp(x_start, -1., 1.)

        if time_next < 0:
            img = x_start
            return img
        
        alpha = self.alphas_cumprod[time]
        alpha_next = self.alphas_cumprod[time_next]

        sigma = self.ddim_sampling_eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
        c = (1 - alpha_next - sigma ** 2).sqrt()

        noise = torch.randn_like(x_start)

        img = x_start * alpha_next.sqrt() + c * model_output + sigma * noise
        
        return img
    
    # @torch.no_grad()
    def ddim_sample(self, params, shape, cond_scale = 3., noise=None, classes=None, conditioning_override=None, clip_denoised = True, **kwargs):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        all_noises = None
        if noise.shape[0] == sampling_timesteps:
            # noise is provided for all timesteps
            all_noises = noise[1:]
            img = noise[0].unsqueeze(0)
        else:
            if noise is None:
                img = torch.randn(shape, device = device)
            else:
                img = noise

        x_start = None
        ctr = 0
        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            # tracker.track()
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, params, cond_scale = cond_scale, clip_x_start = clip_denoised, classes=classes, conditioning_override=conditioning_override, **kwargs)
            # clip_denoised = False

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            if all_noises is not None:
                noise = all_noises[ctr]
            else:
                noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                c * pred_noise + \
                sigma * noise
            
            ctr += 1
        
        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def sample(self, params, cond_scale = 3., classes=None, **kwargs):
        batch_size, image_size, channels = params.shape[0], self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn(params, (batch_size, channels, image_size, image_size), cond_scale, classes=classes, **kwargs)

    def offset_noise(self, x):
        return torch.randn_like(x) + 0.1 * torch.randn(x.shape[0], x.shape[1], 1, 1).to(x.device)

    def q_sample(self, x_start, t, noise=None):
        # noise = default(noise, lambda: torch.randn_like(x_start))
        noise = default(noise, self.offset_noise(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, x_start, t, *, noise = None, classes = None, substance_params=None, return_eps=False, **kwargs):
        noise = default(noise, self.offset_noise(x_start))

        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        t_emb, cond_emb = self.model.preproc_conditioning(t, substance_params, classes)
        model_out = self.model(x, t_emb, cond_emb)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        if return_eps:
            return model_out
        
        loss = self.loss_fn(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean') # (B,)
        loss = loss * extract(self.loss_weight, t, loss.shape)
        # NOTE: this loss is *not* fully reduced (i.e. via mean) for the sake of better loss visualization
        return loss

    def forward(self, img, *args, **kwargs):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        img = self.normalize(img)
        return self.p_losses(img, t, *args, **kwargs), t
    
def create_diffusion_model(config):
    from utils.helpers import count_parameters
    from network.unet import Unet
    import config.noise_config as noise_config

    configs = {
        'extra_tiny': {
            'unet_dim': 32,
            'unet_ch_mult': (1, 2, 4),
            'cond_dim': 128,
            'attention': False
        },
        'tiny': { # ~6.5M params
            'unet_dim': 32,
            'unet_ch_mult': (1, 2, 2, 4),
            'cond_dim': 128
        },
        'medium': { # ~22.5M params
            'unet_dim': 64,
            'unet_ch_mult': (1, 2, 2, 4),
            'cond_dim': 128
        },
        'large': { # ~65M params
            'unet_dim': 64,
            'unet_ch_mult': (1, 2, 4, 8),
            'cond_dim': 128
        }
    }

    model_config = configs[config.model_config]

    nparams = len(noise_config.param_names) # number of noise parameters
    nclasses = len(noise_config.noise_types) # number of noise classes
    
    model = Unet(
        dim=model_config['unet_dim'],
        dim_mults=model_config['unet_ch_mult'],
        channels=1, # grayscale
        cond_dim=model_config['cond_dim'],
        nclasses=nclasses,
        nparams=nparams,
        attention=config.attention,
        pos_enc_deg=config.pos_enc,
        cond_layers=config.cond_layers
    )

    diffusion = GaussianDiffusion(
        model,
        image_size = config.image_size,
        timesteps = config.train_timesteps,
        loss_type = config.loss_fn,
        sampling_timesteps=config.sample_timesteps,
        objective=config.objective
    )

    print('Num model parameters', count_parameters(diffusion))
    return diffusion

def load_diffusion_model(config, device='cpu'):
    result_dir = lambda x : os.path.join(config.out_dir, config.exp_name, x)

    checkpoints = [x for x in os.listdir(os.path.join(config.out_dir, config.exp_name)) if x.startswith('model-')]
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split('-')[1].split('.')[0]))
    checkpoint = checkpoints[-1]
    checkpoint_name = os.path.basename(checkpoint)

    config.checkpoint = checkpoint_name.split('.')[0]

    print(f'Loading checkpoint {checkpoint_name} ...')
    checkpoint = torch.load(result_dir(checkpoint_name), map_location='cpu')

    diffusion = create_diffusion_model(config)
    diffusion.load_state_dict(checkpoint['model'])
    
    print('Model loaded.')

    diffusion.eval()
    diffusion.to(device)

    diffusion.ddim_sampling_eta = 0.0 # deterministic sampling
    diffusion.sampling_timesteps = int(config.sample_timesteps)

    return diffusion