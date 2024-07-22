import torch
import numpy as np

tight_sigmoid = lambda x, s=10 : 1. / (1. + torch.exp(-s*(x - 0.5)))
smoothstep = lambda x: x * x * (3 - 2 * x)
smootherstep = lambda x: x * x * x * (x * (x * 6 - 15) + 10)
smoothstep_6 = lambda x: 924*x**13 - 6006*x**12 + 16380*x**11 - 24024*x**10 + 20020*x**9 - 9009*x**8 + 1716*x**7

def smooth_linear_gradient(W, kernel_width=128, blur_iter=1024):
    def blur_gradient(gradient, width, iterations, kernel_size=5, sigma=1.0):
        gradient = gradient.view(1, 1, -1)
        
        # Create a 1D Gaussian kernel
        kernel = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float32)
        kernel = torch.exp(-kernel**2 / (2 * sigma**2))
        kernel /= kernel.sum()  # Normalize the kernel
        kernel = kernel.view(1, 1, -1)  # Reshape to (1, 1, kernel_size)
        
        # Apply Gaussian blur iteratively
        for _ in range(iterations):
            gradient = torch.nn.functional.pad(gradient, (kernel_size//2, kernel_size//2), mode='reflect')
            gradient = torch.nn.functional.conv1d(gradient, kernel)
        
        return gradient.view(-1)

    linear_gradient = torch.linspace(-1, 1, W)
    linear_gradient = blur_gradient(linear_gradient, width=kernel_width, iterations=blur_iter)
    linear_gradient = (linear_gradient + 1) / 2 # back to [0, 1]
    return linear_gradient

def bilinear_interpolation(d, x):
    """
    Perform bilinear interpolation on a square with four vector valued quantities at its corners.

    :param d: Data tensor of shape (4, C), representing four vectors at the four corners of the square.
    :param x: Coordinates tensor of shape (N, 2), where each row is a pair (u, v) with 0 <= u, v <= 1.
    :return: Interpolated values tensor of shape (N, C).
    """
    d_tl = d[0]  # Top-left
    d_tr = d[2]  # bottom-left
    d_bl = d[1]  # top-right
    d_br = d[3]  # Bottom-right

    u = x[:, 0].unsqueeze(1)
    v = x[:, 1].unsqueeze(1)

    top_interp = (1 - u) * d_tl + u * d_tr
    bottom_interp = (1 - u) * d_bl + u * d_br
    interp_values = (1 - v) * top_interp + v * bottom_interp
    return interp_values

def slerp(val, low, high):
    low_norm = low/torch.norm(low, dim=1, keepdim=True)
    high_norm = high/torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm*high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0-val)*omega)/so).unsqueeze(1)*low + (torch.sin(val*omega)/so).unsqueeze(1) * high
    return res

def complement(x):
    return 1. - x

def tile_codes(z, H, W):
    # given z: (B, C) or (C,) tile to (B, C, H, W)
    if z.ndim == 1:
        z = z.unsqueeze(0)

    z = z.unsqueeze(-1).unsqueeze(-1)
    z = z.expand(-1, -1, H, W)
    return z

def horizontal_blend_map(H, W):
    return torch.linspace(0., 1., W).view(1, 1, 1, W).expand(1, 1, H, W)

def vertical_blend_map(H, W):
    return torch.linspace(0., 1., H).view(1, 1, H, 1).expand(1, 1, H, W)

def square_blend_map(H, W):
    return horizontal_blend_map(H, W) * vertical_blend_map(H, W)

def periodic_horizontal_blend_map(H, W, period = 1.):
    x = torch.cos(horizontal_blend_map(H, W) * period * np.pi)
    x = x * 0.5 + 0.5
    return x

def periodic_vertical_blend_map(H, W, period = 1.):
    x = torch.cos(vertical_blend_map(H, W) * period * np.pi)
    x = x * 0.5 + 0.5
    return x
