import hyperspy.api as hs
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import sys
import yaml
from empanada.inference.engines import PanopticDeepLabEngine

# Add current directory to path to import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import shading_correct_flatfield, denoise_nlm_2d

# Load Configuration
with open('mitonet_config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

# 1. Load Data
# Use path from config, with fallback or override logic if needed
path_slide = cfg['data_loading']['slide_path']

print(f"Loading image from {path_slide}")
img = hs.load(path_slide).data
print(f"Image shape: {img.shape}, dtype: {img.dtype}")

# 2. Preprocess
print("Preprocessing...")

# Shading Correction
if cfg['preprocessing']['shading_correction']['enabled']:
    print("Applying Shading Correction...")
    sc_params = cfg['preprocessing']['shading_correction']
    img, _, _ = shading_correct_flatfield(
        img, 
        method=sc_params['method'], 
        sigma=sc_params['sigma']
    )
# Denoising
if cfg['preprocessing']['denoising']['enabled']:
    print("Applying Denoising...")
    dn_params = cfg['preprocessing']['denoising']
    img, _ = denoise_nlm_2d(
        img, 
        patch_size=dn_params['patch_size'], 
        patch_distance=dn_params['patch_distance'], 
        h_factor=dn_params['h_factor'], 
        fast_mode=dn_params['fast_mode'], 
        eps=1e-6
    )

print("Preprocessing complete.")

# Normalize to [0, 1]
print("Normalizing...")
norm_params = cfg['preprocessing']['normalization']
if img.dtype == np.uint16:
    image = img.astype(np.float32) / 65535.0
elif img.dtype == np.uint8:
    image = img.astype(np.float32) / 255.0
else:
    # Handle arbitrary float or other types with robust scaling
    image = img.astype(np.float32)
    p1 = np.percentile(image, norm_params['percentile_min'])
    p99 = np.percentile(image, norm_params['percentile_max'])
    image = np.clip((image - p1) / (p99 - p1 + 1e-6), 0, 1)

# Convert to tensor (1, 1, H, W)
input_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
print(f"Input tensor shape: {input_tensor.shape}")

# 3. Load Model
model_path = cfg['inference']['model_path']
if not os.path.exists(model_path):
    print(f"Error: Model file {model_path} not found.")
    url = cfg['inference']['model_url']
    print(f"Downloading from {url}...")
    os.system(f"curl -L -O {url}")

print(f"Loading TorchScript model from {model_path}...")
model = torch.jit.load(model_path)
model.eval()

# 4. Inference
print("Starting inference...")
inf_params = cfg['inference']

thing_list = [1] # Mitochondria class
engine = PanopticDeepLabEngine(
    model, 
    thing_list=thing_list, 
    confidence_thr=inf_params['confidence_thr'], 
    nms_threshold=inf_params['nms_threshold'],
    nms_kernel=inf_params['nms_kernel'],
    stuff_area=inf_params['stuff_area']
)

# Device selection
model.to(inf_params['device'])

# Run inference
pan_seg = engine(input_tensor)
print("Inference complete.")

# 5. Visualization
print("Visualizing results...")
vis_params = cfg['output']['visualization']
save_path = cfg['output']['save_path']

mask = pan_seg.cpu().numpy()
print(f"Mask shape before squeeze: {mask.shape}")
mask = mask.squeeze()
print(f"Mask shape after squeeze: {mask.shape}")

fig, ax = plt.subplots(1, 2, figsize=(15, 7))
ax[0].imshow(image, cmap=vis_params['cmap_img'])
ax[0].set_title('Original Image')
ax[0].axis('off')

# Use a distinct colormap for segmentation labels
ax[1].imshow(image, cmap=vis_params['cmap_img'])
ax[1].imshow(mask, cmap=vis_params['cmap_mask'], alpha=vis_params['alpha']) # Overlay
ax[1].set_title('MitoNet Segmentation Overlay')
ax[1].axis('off')

plt.tight_layout()
plt.savefig(save_path)
print(f"Result saved to {save_path}")

