# %%
import hyperspy.api as hs
import numpy as np
import matplotlib.pyplot as plt

# %%
path_data = '/workspaces/mito-counter/data/Etude Calpaine 3/Condition - Calpaine 3'
path_slide = f'{path_data}/TA 1 - WT/TA 1 - ME DATA/TA1-4800X-0002.dm3'

# %%
# load the md3 image
img = hs.load(path_slide)

# %%
# View metadata (very useful for EM data)
print(img.metadata)

# Display the image
img.plot()

# Access the data as a NumPy array
data_array = img.data

# %%
# display the image and the histogram
fig, ax = plt.subplots(1, 2, figsize=(15, 7))
ax[0].imshow(img.data, cmap='gray')
ax[0].set_title('Image')

ax[1].hist(img.data.flatten(), bins=2000)
ax[1].set_title('Intensity histogram')
# ax[1].set_yscale('log')
ax[1].grid()
plt.show()

# %%
print(img.data.shape)


# %%
plt.imshow(img.data < 70000, cmap='gray')
plt.show()


# %%
from utils import shading_correct_flatfield, denoise_nlm_2d

raw_img = img.data
corrected, corrected_u16, illum = shading_correct_flatfield(raw_img, method="gaussian", sigma=100)

# Quick sanity-check plots
fig, ax = plt.subplots(1, 3, figsize=(18, 6))
ax[0].imshow(raw_img, cmap="gray");        ax[0].set_title("Raw")
ax[1].imshow(illum, cmap="gray");     ax[1].set_title("Estimated illumination")
ax[2].imshow(corrected_u16, cmap="gray"); ax[2].set_title("Shading-corrected")
for a in ax: a.axis("off")
plt.show()

# %%
denoised, denoised_u16 = denoise_nlm_2d(corrected, patch_size=5, patch_distance=25, h_factor=20, fast_mode=True, eps=1e-6)

fig, ax = plt.subplots(1, 3, figsize=(20, 6))
ax[0].imshow(raw_img, cmap="gray");        ax[0].set_title("Raw")
ax[1].imshow(corrected_u16, cmap="gray"); ax[1].set_title("Corrected")
ax[2].imshow(denoised_u16, cmap="gray"); ax[2].set_title("Denoised")
for a in ax: a.axis("off")
plt.savefig('/workspaces/mito-counter/data/Etude Calpaine 3/results/denoised.png')
plt.show()
# %%
# Plot the corrected image with histogram
fig, ax = plt.subplots(1, 2, figsize=(18, 6))
ax[0].imshow(corrected, cmap="gray"); ax[0].set_title("Corrected")
ax[1].hist(corrected.flatten(), bins=2000)
ax[1].set_title("Intensity histogram")
ax[1].grid()
plt.savefig('/workspaces/mito-counter/data/Etude Calpaine 3/results/corrected_histogram.png')
plt.show()

# Plot the denoised image with histogram
fig, ax = plt.subplots(1, 2, figsize=(18, 6))
ax[0].imshow(denoised, cmap="gray"); ax[0].set_title("Denoised")
ax[1].hist(denoised.flatten(), bins=2000)
ax[1].set_title("Intensity histogram")
ax[1].grid()
plt.savefig('/workspaces/mito-counter/data/Etude Calpaine 3/results/denoised_histogram.png')
plt.show()
# %%
# Save the corrected image as a tif file
import tifffile as tif
tif.imwrite('/workspaces/mito-counter/data/Etude Calpaine 3/results/raw.tif', raw_img)
tif.imwrite('/workspaces/mito-counter/data/Etude Calpaine 3/results/corrected.tif', corrected)
tif.imwrite('/workspaces/mito-counter/data/Etude Calpaine 3/results/corrected_u16.tif', corrected_u16)
tif.imwrite('/workspaces/mito-counter/data/Etude Calpaine 3/results/illum.tif', illum)
tif.imwrite('/workspaces/mito-counter/data/Etude Calpaine 3/results/denoised.tif', denoised)
tif.imwrite('/workspaces/mito-counter/data/Etude Calpaine 3/results/denoised_u16.tif', denoised_u16)
# %%
from skimage.filters import frangi, threshold_otsu
from skimage.morphology import opening, closing, disk, remove_small_objects
from skimage.measure import label, regionprops
from skimage import exposure

# 1. Enhance Contrast (Optional but helpful for Frangi)
# Rescale intensity to full range for better filtering
p2, p98 = np.percentile(denoised, (2, 98))
img_rescaled = exposure.rescale_intensity(denoised, in_range=(p2, p98))

# 2. Frangi Vesselness Filter
# 'sigmas' determines the scale of tubes to look for.
# Adjust range(1, 10) based on the width of your mitochondria in pixels.
# darker=True looks for dark tubes on bright background (or vice versa).
# Since mitochondria are dark, use black_ridges=True (in newer skimage versions) 
# or invert image if needed.
mito_enhanced = frangi(img_rescaled, sigmas=range(1, 5), black_ridges=True)

# 3. Thresholding the Enhanced Image
# The Frangi filter outputs a probability map (0 to 1). 
# It's much easier to threshold than the raw image.
thresh_val = threshold_otsu(mito_enhanced)
binary_mask = mito_enhanced > thresh_val

# 4. Morphological Cleanup
# Remove noise (small specks) and smooth edges
binary_mask = remove_small_objects(binary_mask, min_size=50) # Adjust min_size
binary_mask = closing(binary_mask, disk(3))

# 5. Label and Count
labeled_mito = label(binary_mask)
regions = regionprops(labeled_mito)

print(f"Counted {len(regions)} mitochondrial structures.")

# Visualization
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(denoised, cmap='gray')
ax[0].set_title('Denoised Input')
ax[1].imshow(mito_enhanced, cmap='magma')
ax[1].set_title('Frangi Vesselness (Tubular Detection)')
ax[2].imshow(labeled_mito, cmap='nipy_spectral')
ax[2].set_title('Final Segmentation')
plt.show()
# %%
