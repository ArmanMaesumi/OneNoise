import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

args = argparse.ArgumentParser()

# Experiment parameters:
args.add_argument('--dry_run', type=str2bool, default=False,        help='Uses a tiny amount of data during training to check if everything is working')
args.add_argument('--model_config', type=str, default='tiny',       help='Model configuration (extra_tiny, tiny, medium, large)')
args.add_argument('--data_dir', type=str, default='./data')
args.add_argument('--out_dir', type=str, default='./results')
args.add_argument('--exp_name', type=str, default=None) # will be auto generated if not provided
args.add_argument('--image_size', type=int, default=256)
args.add_argument('--sample_every', type=int, default=1000,         help='Generate test samples every n steps')
args.add_argument('--sample_timesteps', type=int, default=50,       help='Number of diffusion timesteps to use during testing (DDIM)')
args.add_argument('--train_timesteps', type=int, default=1000,      help='Number of diffusion timesteps to use during training')

args.add_argument('--milestone', type=int, default=None,            help='Resume training from a specific checkpoint')
args.add_argument('--restart_opt', type=str2bool, default=False,    help='Restart optimizer from scratch (only relevant if resuming training)')

args.add_argument('--num_workers', type=int, default=4,             help='Number of workers for data loading')

# Network parameters:
args.add_argument('--attention', type=str2bool, default=True,       help='Use attention layers in U-Net')
args.add_argument('--pos_enc', type=int, default=0,                 help='Positional encoding degree on noise attributes (0=no positional encoding)')
args.add_argument('--cond_layers', type=int, nargs='+', default=[], help='Layer indices for injected conditioning (empty=all layers)')

# Training parameters:
args.add_argument('--batch_size', type=int, default=8,              help='Batch size (per GPU!)')
args.add_argument('--grad_accum', type=int, default=2,              help='Number of batches to accumulate gradients over')
args.add_argument('--lr', type=float, default=8e-5)
args.add_argument('--beta1', type=float, default=0.9)
args.add_argument('--beta2', type=float, default=0.99)
args.add_argument('--ema_decay', type=float, default=0.995)

args.add_argument('--train_num_steps', type=int, default=1_000_000, help='Number of training steps, default is essentially infinite')
args.add_argument('--optim', type=str, default='adamw',             help='Optimizer to use (adam or adamw)')
args.add_argument('--objective', type=str, default='pred_noise',    help='Diffusion training objective (pred_noise, pred_x0)')
args.add_argument('--loss_fn', type=str, default='l2',              help='Loss function (l1, l2)')

args.add_argument('--cutmix', type=int, default=1,                  help='Cutmix: number of applied patches (0=no cutmix)')
args.add_argument('--cutmix_prob', type=float, default=0.5,         help='Cutmix: probability of applying cutmix')
args.add_argument('--cutmix_rot', type=str2bool, default=True,      help='Cutmix: apply random rotation to cutmix masks')

args.add_argument('--emb_penalty', type=float, default=0.02,        help='Lambda for spherical embedding penalty')

args.add_argument('--precision', type=str, default='fp32',          help='Precision to use during training (fp32, fp16, bf16)')
args.add_argument('--tf32', type=str2bool, default=False,           help='Use TensorFloat32 for matmuls, recommended if your GPU supports it')

arguments = args.parse_args()

from config.noise_config import noise_types
arguments.noise_types = noise_types