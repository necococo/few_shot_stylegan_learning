from io import BytesIO
import lmdb
from PIL import Image
from torch.utils.data import Dataset
# import numpy as np
import torch
import random


class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=256):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)

        return img


class AncherDataset(Dataset):# A set of fixed noises to be generated for the anchor region.
    def __init__(self, args, device, n_anc_noise):
        self.device=device
        self.n_anc_noise = n_anc_noise
        self.z_anc_noise = torch.randn(n_anc_noise, args.latent, device=device)#([len(dataset), 512], device=device)
        
    def __getitem__(self, index):
        self.item =self.z_anc_noise[index]
        return self.item

    def __len__(self):
        return  self.n_anc_noise #10

   


