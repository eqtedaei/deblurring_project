import torch
import cv2
from models.msrn_attn import MSRN_Atten

device = "cuda" if torch.cuda.is_available() else "cpu"

model = MSRN_Atten(scales=3, stages=2, base_ch=48).to(device)
model.load_state_dict(torch.load("deblur_model.pth"))
model.eval()

inp_path = "dataset/GoPro/test/blurred/0001.png"
out_path = "result.png"

# load image
img = cv2.imread(inp_path)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
rgb = torch.tensor(rgb).permute(2,0,1).float() / 255.0
rgb = rgb.unsqueeze(0).to(device)

# inference
with torch.no_grad():
    pred = model(rgb)

# back to image format
pred = pred.squeeze().cpu().permute(1,2,0).numpy()
pred = (pred * 255).clip(0,255).astype("uint8")
pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)

cv2.imwrite(out_path, pred)
print("Saved:", out_path)
