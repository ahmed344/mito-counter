import numpy as np
import cv2
from skimage import exposure
from skimage.restoration import denoise_nl_means, estimate_sigma, rolling_ball

def shading_correct_flatfield(raw, method="gaussian", sigma=400, radius=800, eps=1e-6):
    """
    raw: 2D numpy array (e.g. img.data), any dtype
    method:
      - "gaussian": fast, usually good for vignetting
      - "rolling_ball": sometimes more robust but slower
    sigma: blur strength in pixels (bigger = smoother illumination estimate)
    radius: rolling-ball radius in pixels (bigger = smoother)
    """
    raw_f = raw.astype(np.float32)

    if method == "gaussian":
        illum = cv2.GaussianBlur(
            raw_f, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT
        )
    elif method == "rolling_ball":
        illum = rolling_ball(raw_f, radius=radius)
    else:
        raise ValueError("method must be 'gaussian' or 'rolling_ball'")

    # Flat-field correction (division). Scale keeps similar overall brightness.
    scale = np.median(illum[illum > 0]) if np.any(illum > 0) else np.median(illum)
    corrected = raw_f / (illum + eps) * scale

    # Optional: make a nice display/export uint16 image (robust contrast)
    p1, p99 = np.percentile(corrected, (1, 99.8))
    corrected_u16 = exposure.rescale_intensity(
        corrected, in_range=(p1, p99), out_range=(0, 65535)
    ).astype(np.uint16)

    return corrected, corrected_u16, illum


def denoise_nlm_2d(img, patch_size=7, patch_distance=11, h_factor=0.8, fast_mode=True, eps=1e-6):
    """
    Non-Local Means denoising for a 2D image (any dtype).
    h_factor: strength multiplier; higher = smoother (risk: blur fine details)
    patch_size: size of the patch for comparison
    patch_distance: distance between patches
    fast_mode: faster mode for the algorithm
    eps: epsilon for the algorithm
    Returns: denoised_float, denoised_u16
    """
    x = img.astype(np.float32)

    # Robustly scale to [0, 1] for stable parameter behavior
    p1, p99 = np.percentile(x, (1, 99.8))
    x01 = np.clip((x - p1) / (p99 - p1 + eps), 0, 1)

    sigma = float(np.mean(estimate_sigma(x01, channel_axis=None)))
    h = h_factor * sigma

    y01 = denoise_nl_means(
        x01,
        h=h,
        sigma=sigma,
        patch_size=patch_size,
        patch_distance=patch_distance,
        fast_mode=fast_mode,
        channel_axis=None,
    ).astype(np.float32)

    # Back to original-ish scale (float) and a display/export uint16
    y = y01 * (p99 - p1) + p1
    y_u16 = exposure.rescale_intensity(y, in_range=(p1, p99), out_range=(0, 65535)).astype(np.uint16)
    return y, y_u16