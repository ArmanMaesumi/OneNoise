import torch
from torch import nn, einsum
from functools import partial

from .helpers import default, exists
from .blocks import LinearAttention, Attention, Residual, Upsample, Downsample, PreNorm, ResnetBlock, SinusoidalPosEmb, NeRFPosEnc
from .blocks import conv2d

class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        self_condition = False,
        resnet_block_groups = 8,
        learned_variance = False,
        cond_dim = 128,
        nclasses = 1,
        nparams = 18,
        attention = True,
        pos_enc_deg = 0, # default 0 = no pos enc
        cond_layers = (), # which layers to condition, default () = all
    ):
        super().__init__()

        self.nclasses = nclasses
        self.cond_dim = cond_dim

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = conv2d(input_channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        self.cls_dim = dim if nclasses > 1 else 0
        self.classes_emb = nn.Embedding(nclasses, self.cls_dim)

        self.cond_pos_enc = NeRFPosEnc(0, pos_enc_deg) if pos_enc_deg > 0 else nn.Identity()
        cond_enc_dim = (2 * nparams * pos_enc_deg) + nparams # final dim for positonally encoded substance param cond

        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_enc_dim + self.cls_dim, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim)
        )

        sinu_pos_emb = SinusoidalPosEmb(dim)
        fourier_dim = dim

        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)
        if len(cond_layers) == 0:
            cond_layers = list(range(num_resolutions + 2)) # condition on all layers, plus bottleneck and final conv blocks

        if attention:
            atten_block = lambda dim, linear: Residual(PreNorm(dim, LinearAttention(dim))) if linear else Residual(PreNorm(dim, Attention(dim)))
        else:
            atten_block = lambda dim, linear: nn.Identity()

        model_str = 'UNET: '
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            maybe_cond = cond_dim if ind in cond_layers else None # condition only on certain layers

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim, z_emb_dim=maybe_cond),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim, z_emb_dim=maybe_cond),
                atten_block(dim_in, linear=True),
                Downsample(dim_in, dim_out) if not is_last else conv2d(dim_in, dim_out, 3, padding = 1)
            ]))
            model_str += f'down_cond{maybe_cond}_{ind}({dim_in} -> {dim_out}) '

        maybe_cond = cond_dim if num_resolutions in cond_layers else None
        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim, z_emb_dim=maybe_cond)
        self.mid_attn = atten_block(mid_dim, linear=False)
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim, z_emb_dim=maybe_cond)
        model_str += f'mid_cond{maybe_cond}_({mid_dim} -> {mid_dim}) '

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            maybe_cond = cond_dim if (num_resolutions - ind - 1) in cond_layers else None

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim, z_emb_dim=maybe_cond),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim, z_emb_dim=maybe_cond),
                atten_block(dim_out, linear=True),
                Upsample(dim_out, dim_in) if not is_last else  conv2d(dim_out, dim_in, 3, padding = 1)
            ]))
            model_str += f'up_cond{maybe_cond}_{ind}({dim_out} -> {dim_in}) '

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        maybe_cond = cond_dim if num_resolutions + 1 in cond_layers else None
        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim, z_emb_dim=maybe_cond)
        self.final_conv = conv2d(dim, self.out_dim, 1)
        model_str += f'final_cond{maybe_cond}_({dim} -> {self.out_dim})'

        print(model_str)

    def preproc_conditioning(self, time, params, classes, class_emb=None):
        '''
        Preprocesses the diffusion time and conditioning features.

        time:  (B,) tensor of diffusion time values
        params: (B, nparams, H, W) tensor of noise parameters
        classes: (B, C, H, W) tensor of one hot class labels
        class_emb (optional): (B, C, H, W) tensor of class embeddings, if provided `classes` is ignored

        Returns:
        time: (B, time_dim) tensor of processed time embeddings
        c: (B, cond_dim, H, W) tensor of processed conditioning features
        '''

        if exists(class_emb):
            cls_embs = class_emb
        else:
            cls_embs = einsum('b c h w, c d -> b d h w', classes, self.classes_emb.weight)
        c = torch.cat((params, cls_embs), dim = 1) # (B, C' + cls_emb, H, W)
        c = torch.permute(c, (0, 2, 3, 1)) # (B, H, W, C' + cls_emb)
        c = self.cond_mlp(c)
        c = c.permute(0, 3, 1, 2) # back to (B, C, H, W)
        c = c.contiguous()

        time = self.time_mlp(time)
        
        return time, c

    # @torch.compile()
    def forward(
        self,
        x,       # current image (B, 1, H, W)
        t,       # diffusion time (B,)
        c,       # conditioning feature grid (B, C, H, W)
    ):
        x = self.init_conv(x)
        r = x.clone()

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t, c)
            h.append(x)
            x = block2(x, t, c)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t, c)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t, c)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t, c)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t, c)
            x = attn(x)
            x = upsample(x)

        x = torch.cat((x, r), dim = 1)
        x = self.final_res_block(x, t, c)
        x = self.final_conv(x)

        return x