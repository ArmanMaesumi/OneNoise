import json
import math
import torch
import wandb

from pathlib import Path
from tqdm.auto import tqdm
from ema_pytorch import EMA
from accelerate import Accelerator
from torchvision.utils import save_image, make_grid

from inference.inference import Inference
from inference.example_noises import horizontal_blends, vertical_blends
from noise_data import HDF5Dataset

def exists(x):
    return x is not None

def cycle(dl):
    while True:
        for data in dl:
            yield data

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        config,
        *,
        train_batch_size=16,
        gradient_accumulate_every=1,
        train_num_steps=100000,
        ema_update_every=10,
        ema_decay=0.995,
        results_folder='./results',
        split_batches=True,
        precision='fp32',
    ):
        super().__init__()
        
        self.config = config

        assert precision in ['fp32', 'fp16', 'bf16']

        print('Training with ', precision, ' precision')

        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision = 'no' if precision == 'fp32' else precision,
            # kwargs_handlers=[
            #     InitProcessGroupKwargs(timeout=3600) # try to avoid strange timeout errors
            # ]
        )

        if self.accelerator.is_main_process:
            wandb.init(
                mode="online" if not config.dry_run else "disabled",
                project="one-noise",
                config=config,  
            )

        self.model = diffusion_model

        self.sample_every = config.sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps

        world_size = self.accelerator.num_processes
        rank = self.accelerator.process_index
        print(f"world_size: {world_size}, rank: {rank}")

        self.create_dataloader(config, rank, world_size, train_batch_size)
        self.create_optimizer(config)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)

        self.ema = None
        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)
            self.inference = Inference(
                config,
                model=self.ema,
                device=self.accelerator.device,
                save_dir=None,
                seed=None
            )

        self.step = 0

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

        # save config as json:
        with open(self.results_folder / 'config.json', 'w') as f:
            json.dump(config.__dict__, f, indent=2)

    def create_optimizer(self, config):
        if config.optim == 'adam':
            self.opt = torch.optim.Adam(self.model.parameters(), lr=config.lr, betas=(config.beta1, config.beta2))
        elif config.optim == 'adamw':
            self.opt = torch.optim.AdamW(self.model.parameters(), lr=config.lr, betas=(config.beta1, config.beta2))

    def create_dataloader(self, config, rank, world_size, batch_size):
        '''
        Split the dataset across all processes. Since the dataset is large this should be fine.
        '''
        if config.dry_run: # This simulates a small dataset for debugging purposes
            world_size = 4096

        self.ds = HDF5Dataset(
            noise_types=config.noise_types,
            data_dir=config.data_dir,
            augment=True,
            cutmix=config.cutmix,
            cutmix_prob=config.cutmix_prob,
            cutmix_rot=config.cutmix_rot,
            rank=rank,
            world_size=world_size,
        )

        self.dl = torch.utils.data.DataLoader(
            self.ds,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=config.num_workers // world_size,
            drop_last=True
        )

        print(f"dataloader: {len(self.dl)}")
        print(f"dataset: {len(self.ds)}")
        self.dl = cycle(self.dl)


    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone, restart_optimizer=False):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']

        if not restart_optimizer:
            self.opt.load_state_dict(data['opt'])

        if exists(self.ema):
            self.ema.load_state_dict(data['ema'])

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device
        
        # Spherical regularization for class embeddings:
        emb_lambda = self.config.emb_penalty
        penalize_emb = self.config.emb_penalty > 0.0
        
        model_ptr = self.model.module.model if hasattr(self.model, 'module') else self.model.model
        if penalize_emb:
            # grab the mean magnitude of initialized embeddings -- w/ initialization of N^32(0,1) the expected norm is about sqrt(32) ~= 5.65
            emb_target_norm = model_ptr.classes_emb.weight.norm(dim=-1).mean().detach().item()
            print(f"embedding target norm: {emb_target_norm:.4f}")
        
        # For logging loss in timestep bins:
        bin_size = 100
        nbins = 1000 // bin_size
        total_loss_bins = torch.zeros(nbins)

        # Some hard-coded noise parameters to test the model's ability to interpolate between classes:
        vertical_strips = vertical_blends()
        horizontal_strips = horizontal_blends()

        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:
            while self.step < self.train_num_steps:
                total_loss = 0.
                total_emb_penalty = 0.
                total_loss_bins.zero_()

                for _ in range(self.gradient_accumulate_every):
                    imgs, noise_classes, noise_params = next(self.dl)
                    imgs = imgs.to(device)

                    noise_classes = noise_classes.to(device)
                    noise_params = noise_params.to(device)

                    with self.accelerator.autocast():
                        losses, ts = self.model(imgs, substance_params=noise_params, classes=noise_classes)
                        losses = losses.mean(dim=-1) # (B,)

                        # We will visualize losses in timestep bins of size 100:
                        bin_idx = torch.div(ts.detach().cpu(), bin_size, rounding_mode='trunc')
                        total_loss_bins.scatter_add_(0, bin_idx, losses.detach().cpu() / self.gradient_accumulate_every / len(losses))

                        loss = losses.mean() / self.gradient_accumulate_every
                        
                        if penalize_emb: # TODO: this should probably be moved outside the gradient accumulation loop
                            emb_norms = model_ptr.classes_emb.weight.norm(dim=-1)
                            emb_loss = emb_lambda * (emb_norms - emb_target_norm).pow(2).mean() / self.gradient_accumulate_every

                            loss += emb_loss
                            total_emb_penalty += emb_loss.item()

                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)

                pbar.set_description(f'loss: {total_loss:.4f}')
                if self.accelerator.is_main_process:
                    wandb.log({'loss': total_loss, 'l2_loss': total_emb_penalty})
                    for i in range(nbins):
                        wandb.log({f'loss_t = {i * bin_size} - {(i + 1) * bin_size}' : total_loss_bins[i].item()})

                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.to(device)
                    self.ema.update()

                    if self.step != 0 and self.step % self.sample_every == 0:
                        self.ema.ema_model.eval()

                        with torch.no_grad():
                            milestone = self.step // self.sample_every

                            full_grid = self.inference.full_grid(256, 256, num_samples=4, filename=f'full_grid-{milestone}')

                            imgs = self.inference.random_class_interpolations(512, 512, nimg=16, filename=f'slerp512-{milestone}')
                            grid1 = make_grid(imgs, nrow=int(math.sqrt(len(imgs))))
                            wandb.log({"slerp512": [wandb.Image(grid1, caption=f'step {self.step}')], "full_grid": [wandb.Image(full_grid, caption=f'step {self.step}')]})
                            # imgs = self.inference.random_class_interpolations(1024, 1024, nimg=4, filename=f'slerp1024-{milestone}')
                            # grid2 = make_grid(imgs, nrow=int(math.sqrt(len(imgs))))
                            # wandb.log({"slerp512": [wandb.Image(grid1, caption=f'step {self.step}')], "slerp1024": [wandb.Image(grid2, caption=f'step {self.step}')]})

                            vertical_outs = []
                            for noise1, noise2 in vertical_strips:
                                vertical_outs += [ self.inference.slerp_mask(mask='./inference/masks/linear-wipe-down.png',
                                                                            blending_factor=1.,
                                                                            dict1=noise1,
                                                                            dict2=noise2,
                                                                            H=1024,
                                                                            W=256) ]
                            vertical_outs = torch.cat(vertical_outs, dim=0)
                            vertical_grid = make_grid(vertical_outs, nrow=int(math.sqrt(len(vertical_outs))), padding=10)

                            horizontal_outs = []
                            for noise1, noise2 in horizontal_strips:
                                horizontal_outs += [ self.inference.slerp_mask(mask='./inference/masks/linear-wipe-right.png',
                                                                            blending_factor=1.,
                                                                            dict1=noise1,
                                                                            dict2=noise2,
                                                                            H=256,
                                                                            W=1024) ]
                            horizontal_outs = torch.cat(horizontal_outs, dim=0)
                            horizontal_grid = make_grid(horizontal_outs, nrow=int(math.sqrt(len(horizontal_outs))), padding=10)
                            
                            wandb.log({"horizontal_strips": [wandb.Image(horizontal_grid, caption=f'step {self.step}')],
                                       "vertical_strips": [wandb.Image(vertical_grid, caption=f'step {self.step}')]})
                            save_image(horizontal_grid, self.results_folder / f'out/horizontal_strips-{milestone}.png')
                            save_image(vertical_grid, self.results_folder / f'out/vertical_strips-{milestone}.png')

                            self.save(milestone)

                        print(f'loss: {total_loss:.4f}')

                pbar.update(1)

        accelerator.print('training complete')
