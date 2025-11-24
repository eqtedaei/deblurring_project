import os
import cv2
import torch
from torch.utils.data import Dataset

class DeblurDataset(Dataset):
    def __init__(self, blurred_dir, sharp_dir, crop=256):
        self.blurred_dir = blurred_dir
        self.sharp_dir = sharp_dir
        self.files = os.listdir(blurred_dir)
        self.crop = crop

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        bl = cv2.imread(os.path.join(self.blurred_dir, name))
        sh = cv2.imread(os.path.join(self.sharp_dir, name))

        bl = cv2.cvtColor(bl, cv2.COLOR_BGR2RGB)
        sh = cv2.cvtColor(sh, cv2.COLOR_BGR2RGB)

        bl = cv2.resize(bl, (self.crop, self.crop))
        sh = cv2.resize(sh, (self.crop, self.crop))

        bl = torch.tensor(bl).permute(2,0,1).float() / 255.0
        sh = torch.tensor(sh).permute(2,0,1).float() / 255.0
        return bl, sh
