import os
import torch

from trainer import Trainer
from datetime import datetime
from network.diffusion import create_diffusion_model

def run(config):
    print(config)

    if config.tf32:
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True 

    diffusion = create_diffusion_model(config)

    # if we are not resuming from a checkpoint, generate a new experiment name
    if config.exp_name is None and config.milestone is None:
        config.exp_name = f'{config.model_config}_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}'

    result_dir = os.path.join(config.out_dir, config.exp_name)

    trainer = Trainer(
        diffusion,
        config,
        results_folder=result_dir,
        train_batch_size=config.batch_size,
        train_num_steps=config.train_num_steps,
        gradient_accumulate_every=config.grad_accum,
        ema_decay=config.ema_decay,
        split_batches=False,
        precision=config.precision,
    )

    if config.milestone is not None:
        trainer.load(config.milestone)

    trainer.train()
    print('Training complete.')

if __name__ == '__main__':
    from args import arguments
    
    run(arguments)
