import os
import torch
import numpy as np
import cv2

from pathlib import Path

def video_tensor_preprocess(tensor):
    if isinstance(tensor, torch.Tensor):
        if tensor.dtype == torch.uint8:
            return tensor
        tensor = tensor.detach().cpu().numpy()
    elif tensor.dtype == np.uint8:
        return tensor
    
    chn_last = tensor.shape[-1] == 3 or tensor.shape[-1] == 1
    chn_dim = -1 if chn_last else 1

    if tensor.shape[chn_dim] == 1: # greyscale
        tensor = np.repeat(tensor, 3, axis=chn_dim)
    
    if not chn_last:
        tensor = tensor.transpose(0, 2, 3, 1)
    
    tensor = tensor * 255.
    tensor = tensor.astype(np.uint8)
    return tensor

def save_tensor_as_video(tensor, filename, fps=30, reencode=False):
    """
    Saves a NumPy tensor as an MP4 video file.

    :param tensor: NumPy tensor of shape (N, H, W, 3)
    :param filename: Name of the output MP4 file
    :param fps: Frames per second for the output video
    """
    tensor = video_tensor_preprocess(tensor)

    if len(tensor.shape) != 4 or tensor.shape[3] != 3:
        raise ValueError("Tensor must have shape (N, H, W, 3)")

    N, H, W, _ = tensor.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename + '.mp4', fourcc, fps, (W, H))

    for i in range(N):
        frame = tensor[i].astype(np.uint8)
        out.write(frame)

    out.release()

    if reencode:
        # Sometimes the output video is not compatible with some video players (vscode)
        # reencode with FFMPEG to fix it:
        fname = Path(filename + '.mp4')
        os.system(f'env -i ffmpeg -i {fname} -c:v libx264 -crf 25 -preset slow -c:a copy {fname}_reencoded.mp4 -y')
        os.remove(fname)
        os.rename(f'{fname}_reencoded.mp4', fname)