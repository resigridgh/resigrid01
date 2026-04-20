from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def denorm_to_01(x: torch.Tensor) -> torch.Tensor:
    """
    Convert from [-1,1] to [0,1] and sanitize invalid values.
    """
    x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
    return torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)


def rgb_to_gray_batch(x: torch.Tensor) -> torch.Tensor:
    """
    x: (B,3,H,W) in [0,1]
    returns: (B,1,H,W)
    """
    x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
    x = torch.clamp(x, 0.0, 1.0)

    r = x[:, 0:1]
    g = x[:, 1:2]
    b = x[:, 2:3]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    gray = torch.nan_to_num(gray, nan=0.0, posinf=1.0, neginf=0.0)
    gray = torch.clamp(gray, 0.0, 1.0)
    return gray


def _conv2d_same(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    x: (B,1,H,W)
    kernel: (1,1,k,k)
    """
    k = kernel.shape[-1]
    pad = k // 2
    return F.conv2d(x, kernel, padding=pad)


def variance_of_laplacian(gray: torch.Tensor) -> torch.Tensor:
    gray = torch.nan_to_num(gray, nan=0.0, posinf=1.0, neginf=0.0)
    gray = torch.clamp(gray, 0.0, 1.0)

    lap_kernel = torch.tensor(
        [[0, 1, 0],
         [1, -4, 1],
         [0, 1, 0]],
        dtype=gray.dtype,
        device=gray.device,
    ).view(1, 1, 3, 3)

    lap = _conv2d_same(gray, lap_kernel)
    lap = torch.nan_to_num(lap, nan=0.0, posinf=0.0, neginf=0.0)

    b = lap.shape[0]
    lap_flat = lap.view(b, -1)
    mean = lap_flat.mean(dim=1, keepdim=True)
    var = ((lap_flat - mean) ** 2).mean(dim=1)
    var = torch.nan_to_num(var, nan=0.0, posinf=0.0, neginf=0.0)
    return var


def tenengrad(gray: torch.Tensor) -> torch.Tensor:
    gray = torch.nan_to_num(gray, nan=0.0, posinf=1.0, neginf=0.0)
    gray = torch.clamp(gray, 0.0, 1.0)

    sobel_x = torch.tensor(
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]],
        dtype=gray.dtype,
        device=gray.device,
    ).view(1, 1, 3, 3)

    sobel_y = torch.tensor(
        [[-1, -2, -1],
         [0, 0, 0],
         [1, 2, 1]],
        dtype=gray.dtype,
        device=gray.device,
    ).view(1, 1, 3, 3)

    gx = _conv2d_same(gray, sobel_x)
    gy = _conv2d_same(gray, sobel_y)
    gx = torch.nan_to_num(gx, nan=0.0, posinf=0.0, neginf=0.0)
    gy = torch.nan_to_num(gy, nan=0.0, posinf=0.0, neginf=0.0)

    mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)
    mag = torch.nan_to_num(mag, nan=0.0, posinf=0.0, neginf=0.0)

    b = mag.shape[0]
    mag_flat = mag.view(b, -1)
    mean = mag_flat.mean(dim=1, keepdim=True)
    var = ((mag_flat - mean) ** 2).mean(dim=1)
    var = torch.nan_to_num(var, nan=0.0, posinf=0.0, neginf=0.0)
    return var


def high_frequency_energy_ratio(gray: torch.Tensor, alpha: float = 0.15) -> torch.Tensor:
    """
    Uses magnitude spectrum ratio outside low-frequency center disc.
    """
    gray = torch.nan_to_num(gray, nan=0.0, posinf=1.0, neginf=0.0)
    gray = torch.clamp(gray, 0.0, 1.0)

    b, _, h, w = gray.shape
    gray_np = gray.detach().cpu().numpy()
    vals = []

    for i in range(b):
        img = np.nan_to_num(gray_np[i, 0], nan=0.0, posinf=1.0, neginf=0.0)
        img = np.clip(img, 0.0, 1.0)

        fft = np.fft.fft2(img)
        fft_shift = np.fft.fftshift(fft)
        mag = np.abs(fft_shift)
        mag = np.nan_to_num(mag, nan=0.0, posinf=0.0, neginf=0.0)

        cy, cx = h // 2, w // 2
        r = int(alpha * min(h, w))

        yy, xx = np.ogrid[:h, :w]
        mask_low = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
        high_energy = mag[~mask_low].sum()
        total_energy = mag.sum()

        if not np.isfinite(total_energy) or total_energy <= 1e-12:
            vals.append(0.0)
        else:
            vals.append(float(high_energy / total_energy))

    return torch.tensor(vals, dtype=gray.dtype, device=gray.device)


def mean_local_std(gray: torch.Tensor, window_size: int = 7) -> torch.Tensor:
    """
    MLSD using box filtering.
    """
    gray = torch.nan_to_num(gray, nan=0.0, posinf=1.0, neginf=0.0)
    gray = torch.clamp(gray, 0.0, 1.0)

    pad = window_size // 2
    mean = F.avg_pool2d(gray, kernel_size=window_size, stride=1, padding=pad)
    mean_sq = F.avg_pool2d(gray ** 2, kernel_size=window_size, stride=1, padding=pad)
    var = torch.clamp(mean_sq - mean ** 2, min=0.0)
    std = torch.sqrt(var + 1e-8)
    std = torch.nan_to_num(std, nan=0.0, posinf=0.0, neginf=0.0)
    return std.view(std.shape[0], -1).mean(dim=1)


def _glcm_contrast_single(
    img_gray_01: np.ndarray,
    levels: int = 16,
    dx: int = 1,
    dy: int = 0,
) -> float:
    """
    Simple GLCM contrast for one grayscale image in [0,1].
    """
    img_gray_01 = np.nan_to_num(img_gray_01, nan=0.0, posinf=1.0, neginf=0.0)
    img_gray_01 = np.clip(img_gray_01, 0.0, 1.0)

    img_q = np.floor(img_gray_01 * (levels - 1)).astype(np.int32)
    img_q = np.clip(img_q, 0, levels - 1)

    h, w = img_q.shape
    glcm = np.zeros((levels, levels), dtype=np.float64)

    for y in range(h):
        ny = y + dy
        if ny < 0 or ny >= h:
            continue
        for x in range(w):
            nx = x + dx
            if nx < 0 or nx >= w:
                continue
            i = img_q[y, x]
            j = img_q[ny, nx]
            glcm[i, j] += 1.0

    glcm = glcm + glcm.T
    total = glcm.sum()
    if total <= 1e-12 or not np.isfinite(total):
        return 0.0

    p = glcm / total
    contrast = 0.0
    for i in range(levels):
        for j in range(levels):
            contrast += (i - j) ** 2 * p[i, j]

    if not np.isfinite(contrast):
        return 0.0

    return float(contrast)


def glcm_contrast(gray: torch.Tensor, levels: int = 16) -> torch.Tensor:
    gray = torch.nan_to_num(gray, nan=0.0, posinf=1.0, neginf=0.0)
    gray = torch.clamp(gray, 0.0, 1.0)

    b = gray.shape[0]
    gray_np = gray.detach().cpu().numpy()
    vals = []

    for i in range(b):
        img = np.nan_to_num(gray_np[i, 0], nan=0.0, posinf=1.0, neginf=0.0)
        img = np.clip(img, 0.0, 1.0)
        vals.append(_glcm_contrast_single(img, levels=levels))

    return torch.tensor(vals, dtype=gray.dtype, device=gray.device)


def compute_all_metrics(images: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    images: (B,3,H,W), expected in [-1,1] or [0,1]
    """
    images = torch.nan_to_num(images, nan=0.0, posinf=1.0, neginf=-1.0)

    if images.min() < 0:
        imgs_01 = denorm_to_01(images)
    else:
        imgs_01 = torch.clamp(images, 0.0, 1.0)

    imgs_01 = torch.nan_to_num(imgs_01, nan=0.0, posinf=1.0, neginf=0.0)
    imgs_01 = torch.clamp(imgs_01, 0.0, 1.0)

    gray = rgb_to_gray_batch(imgs_01)
    gray = torch.nan_to_num(gray, nan=0.0, posinf=1.0, neginf=0.0)
    gray = torch.clamp(gray, 0.0, 1.0)

    return {
        "VoL": variance_of_laplacian(gray),
        "Tenengrad": tenengrad(gray),
        "HFE": high_frequency_energy_ratio(gray),
        "MLSD": mean_local_std(gray),
        "GLCM_Contrast": glcm_contrast(gray),
    }


def aggregate_metric_dict(metric_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
    out = {}
    for k, v in metric_dict.items():
        val = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).mean().item()
        out[k] = float(val)
    return out


def save_metric_barplot(results: Dict[str, Dict[str, float]], save_path: str):
    """
    results = {
        "vae": {"VoL":..., "Tenengrad":..., ...},
        "gan": {...},
        "diffusion": {...}
    }
    """
    model_names = list(results.keys())
    metric_names = list(next(iter(results.values())).keys())

    x = np.arange(len(metric_names))
    width = 0.25

    plt.figure(figsize=(12, 5))

    for idx, model_name in enumerate(model_names):
        vals = [results[model_name][m] for m in metric_names]
        vals = [0.0 if not np.isfinite(v) else v for v in vals]
        plt.bar(x + idx * width, vals, width=width, label=model_name)

    plt.xticks(x + width, metric_names, rotation=20)
    plt.ylabel("Average Metric Value")
    plt.title("Image Quality Comparison Across Generative Models")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_image_grid(images: torch.Tensor, save_path: str, nrow: int = 5):
    from torchvision.utils import save_image

    images = torch.nan_to_num(images, nan=0.0, posinf=1.0, neginf=-1.0)
    imgs = denorm_to_01(images)
    imgs = torch.nan_to_num(imgs, nan=0.0, posinf=1.0, neginf=0.0)
    imgs = torch.clamp(imgs, 0.0, 1.0)
    save_image(imgs, save_path, nrow=nrow)
