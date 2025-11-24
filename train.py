import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.msrn_attn import MSRN_Atten
from utils import DeblurDataset
from tqdm import tqdm

# parameters
EPOCHS = 50
LR = 1e-4
BATCH = 4

device = "cuda" if torch.cuda.is_available() else "cpu"

# dataset paths
train_blur = "dataset/PASCAL/train/blurred"
train_sharp = "dataset/PASCAL/train/sharp"

# load dataset
train_ds = DeblurDataset(train_blur, train_sharp)
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)

# model
model = MSRN_Atten(scales=3, stages=2, base_ch=48).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.L1Loss()

print("Training started ...")

for epoch in range(EPOCHS):
    loop = tqdm(train_loader)
    for blurred, sharp in loop:
        blurred = blurred.to(device)
        sharp = sharp.to(device)

        pred = model(blurred)

        loss = criterion(pred, sharp)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_description(f"Epoch {epoch+1}/{EPOCHS}")
        loop.set_postfix(loss=loss.item())

torch.save(model.state_dict(), "deblur_model.pth")
print("Training finished! Model saved.")
