conda create -n onenoise python=3.11
conda activate onenoise
conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install ema-pytorch wandb tqdm accelerate scipy h5py
pip install "einops>=0.6.1"