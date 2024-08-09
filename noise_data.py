import os
import h5py
import torch
import random
import numpy as np
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset

@torch.no_grad()
def cutmix_augmentation(image, label, sbs_params, niters, dataset, rot=True):
    '''
    Applies cutmix augmentation to the input image by sampling other images in the dataset
    and pasting their crops onto the input image.

    image:      (1, H, W) noise texture
    label:      integer class label
    sbs_params: (num_params,) tensor of noise attributes
    niters:     integer number of cutmix patches to apply to `image`
    dataset:    HDF5Dataset object, used to sample other images randomly
    rot:        whether to rotate the cutmix mask

    Returns:
    Three spatial tensors:  new_image, label, sbs_params
    new_image:              (1, H, W) tensor of `image` with cutmix patches applied
    label:                  (num_classes, H, W) tensor of one-hot encoded class labels at each pixel
    sbs_params:             (num_params, H, W) tensor of noise attributes at each pixel
    '''
    H, W = image.shape[1:]
    n_chn = 2 + sbs_params.shape[0] # total number of channels (includes: image + label + num params)
    data_tensor = torch.empty((n_chn, H, W))
    data_tensor[0] = image
    data_tensor[1] = label
    data_tensor[2:] = sbs_params.unsqueeze(1).unsqueeze(2).expand(-1, H, W)

    all_labels = [label.item()] # keep track of which noise types are already used
    niters = random.randint(1, niters) # uniformly sample number of cutmix patches to apply
    for _ in range(niters):
        # Sample another image from the dataset (mutually exclusive!):
        other_label = label
        while other_label in all_labels: # make sure to get a new noise type
            other = random.randint(0, len(dataset)-1)
            other_img, other_label, other_sbs_params = dataset.__getitem__(other, cutmix_off=True)
            other_label = other_label[:, 0, 0].argmax().item()
        all_labels.append(other_label)

        # Sample a mask for the cutmix area
        # TODO: we used a margin of 64 pixels -- using a margin of 0 might be fine
        m = 64
        top_left = torch.randint(0, image.shape[1]//2 - m, (2,))
        bottom_right = torch.tensor([
            random.randint(top_left[0] + m, image.shape[1]),
            random.randint(top_left[1] + m, image.shape[1])
        ])
        bbox_w, bbox_h = bottom_right - top_left
        bbox_w2 = torch.div(bbox_w, 2, rounding_mode='floor')
        bbox_h2 = torch.div(bbox_h, 2, rounding_mode='floor')    
        origin = torch.tensor([
            random.randint(bbox_w2, image.shape[1] - bbox_w2),
            random.randint(bbox_h2, image.shape[1] - bbox_h2)
        ])

        # create mask of cutmix region:
        # if pixel is inside of bbox, then 1, else 0
        inside = torch.abs(torch.arange(image.shape[1]) - origin[0]).unsqueeze(0).expand(image.shape[1], -1) <= bbox_w2
        inside = torch.logical_and(
            inside,
            torch.abs(torch.arange(image.shape[2]) - origin[1]).unsqueeze(1).expand(-1, image.shape[2]) <= bbox_h2
        )
        mask = torch.where(
            inside,
            torch.tensor(1.),
            torch.tensor(0.)
        ) == 1.

        # rotate the mask by a random angle:
        if rot:
            angle = random.random() * 360.
            mask = T.functional.rotate(mask[None], angle, interpolation=T.InterpolationMode.NEAREST, expand=False, center=(origin[0], origin[1]), fill=0.)
            mask = mask.squeeze(0)

        # update data tensor according to the mask:
        data_tensor[0, mask] = other_img[:, mask]
        data_tensor[1, mask] = other_label
        data_tensor[2:, mask] = other_sbs_params[:, mask]

    # one hot encode labels into (num_classes, H, W) tensor:
    labels = data_tensor[1] # (1, H, W) 
    labels = torch.nn.functional.one_hot(labels.long(), num_classes=len(dataset.noise_types)).permute(2, 0, 1).float()
    return data_tensor[0].unsqueeze(0), labels, data_tensor[2:]

@torch.no_grad()
def HDF5_batch_preproc(batch, device, normalize_fn=None):
    # assume to be given a (B, 1, H, W) uint8 tensor, convert to float and move device:
    # https://gist.github.com/xvdp/149e8c7f532ffb58f29344e5d2a1bee0
    batch = batch.to(device=device).to(dtype=torch.float).div_(255.)
    if normalize_fn is not None:
        batch = normalize_fn(batch)
    return batch

class HDF5Dataset(Dataset):
    def __init__(self,
                 noise_types,
                 data_dir='./data',
                 augment=True,
                 cutmix=0,
                 cutmix_prob=0.5,
                 cutmix_rot=True,
                 rank=0,
                 world_size=1,
                 max_samples=None
                ) -> None:
        super().__init__()
        self.H = 256
        self.W = 256 # hardcoded for our dataset
        
        self.noise_types = noise_types
        self.data_dir = data_dir
        self.cutmix = cutmix
        self.cutmix_prob = cutmix_prob
        self.cutmix_rot = cutmix_rot
        self.rank = rank
        self.world_size = world_size

        # look for all files : noise_separate_ptX.hdf5
        ds_list = [f for f in os.listdir(data_dir) if f.startswith('onenoise_dataset_part') and f.endswith('.hdf5')]
        datapacks = [h5py.File(os.path.join(data_dir, f), 'r') for f in ds_list]
        
        nimages_per_type = datapacks[0].attrs['num_images_per_type'] * len(datapacks)
        nimages = len(noise_types) * nimages_per_type

        # separated data format:
        self.data = np.empty((nimages // world_size, self.H, self.W), dtype=np.uint8)
        self.cls_labels = np.empty((nimages // world_size), dtype=np.int32)

        print(self.data.shape)
        print('Loading noise images...')
        for i, ntype in enumerate(noise_types):
            print(f'loading {ntype} (', i+1, '/', len(noise_types), ')')
            print([dp[ntype].shape for dp in datapacks])
            # concat images from all datapacks for each noise type:
            l1 = i * nimages_per_type // world_size
            r1 = (i+1) * nimages_per_type // world_size
            l2 = rank * (nimages_per_type // world_size // len(datapacks))
            r2 = (rank+1) * (nimages_per_type // world_size // len(datapacks))
            self.data[l1:r1] = np.concatenate([dp[ntype][l2:r2] for dp in datapacks], axis=0)
            self.cls_labels[i * nimages_per_type // world_size:(i+1) * nimages_per_type // world_size] = i

        for dp in datapacks:
            dp.close()

        self.data = torch.from_numpy(self.data) # will retain uint8 dtype
        self.data = self.data.unsqueeze(1) # add channel dimension
        
        if max_samples is not None:
            self.data = self.data[:max_samples]
            self.cls_labels = self.cls_labels[:max_samples]

        self.cls_labels = torch.from_numpy(self.cls_labels).long()
        self.length = len(self.data)

        random_transforms = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
        ])
        self.augment = random_transforms if augment else nn.Identity()

        self.data_min = self.data.min().to(dtype=torch.float) / 255.
        self.data_max = self.data.max().to(dtype=torch.float) / 255.

        # Labels (already preprocessed):
        sbsparams_list = [f for f in os.listdir(data_dir) if f.startswith('onenoise_dataset_parameters_part') and f.endswith('.hdf5')]
        sbsparams_ds = [h5py.File(os.path.join(data_dir, f), 'r') for f in sbsparams_list]

        self.sbsparams = np.empty((nimages // world_size, sbsparams_ds[0].attrs['num_params']), dtype=np.float32)
        print('Loading noise params...')
        # make sure to use the same ordering of noise types as in the data file:
        for i, ntype in enumerate(noise_types):
            l1 = i * nimages_per_type // world_size
            r1 = (i+1) * nimages_per_type // world_size
            l2 = rank * (nimages_per_type // world_size // len(datapacks))
            r2 = (rank+1) * (nimages_per_type // world_size // len(datapacks))
            self.sbsparams[l1:r1] = np.concatenate([dp[ntype][l2:r2] for dp in sbsparams_ds], axis=0)
    
        for dp in sbsparams_ds:
            dp.close()
        self.sbsparams = torch.from_numpy(self.sbsparams).float()

    def normalize(self, x):
        return (x - self.data_min) / (self.data_max - self.data_min)

    def denormalize(self, x):
        return x * (self.data_max - self.data_min) + self.data_min

    def __len__(self):
        return self.length

    def __getitem__(self, idx, cutmix_off=False):
        imgs = self.augment(self.data[idx])
        imgs = HDF5_batch_preproc(imgs, device='cpu', normalize_fn=self.normalize) # (1, H, W)
        cls_labels = self.cls_labels[idx]   # (1,)
        sbsparams = self.sbsparams[idx]     # (num_params,)
        if self.cutmix > 0 and not cutmix_off and random.random() < self.cutmix_prob:
            return cutmix_augmentation(imgs, cls_labels, sbsparams, self.cutmix, self, rot=self.cutmix_rot)
        else:
            # expand cls_labels and sbsparams to be spatial tensor:
            cls_labels = torch.nn.functional.one_hot(cls_labels.long(), num_classes=len(self.noise_types)).float() # (num_classes,)
            cls_labels = cls_labels.unsqueeze(1).unsqueeze(2).expand(-1, imgs.shape[-2], imgs.shape[-1])
            sbsparams = sbsparams.unsqueeze(1).unsqueeze(2).expand(-1, imgs.shape[-2], imgs.shape[-1])
        
        return imgs, cls_labels, sbsparams
