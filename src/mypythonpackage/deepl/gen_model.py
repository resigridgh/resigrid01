import io
import math
import os
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms


# ============================================================
# UTILS
# ============================================================

def get_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# ============================================================
# DATASET
# ============================================================

class CelebAZipDataset(Dataset):

    def __init__(self, zip_path: str, transform=None):
        self.zip_path = zip_path
        self.transform = transform

        with zipfile.ZipFile(zip_path, "r") as zf:
            self.image_names = sorted(
                [
                    name
                    for name in zf.namelist()
                    if name.lower().endswith((".jpg", ".jpeg", ".png"))
                ]
            )

        if len(self.image_names) == 0:
            raise ValueError(f"No images found inside zip file: {zip_path}")

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int) -> torch.Tensor:
        with zipfile.ZipFile(self.zip_path, "r") as zf:
            with zf.open(self.image_names[idx]) as f:
                img = Image.open(io.BytesIO(f.read())).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img


class CelebAFolderDataset(Dataset):

    def __init__(self, root_dir: str, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform

        self.image_paths = sorted(
            [
                p for p in self.root_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png"]
            ]
        )

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in folder: {root_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.image_paths[idx]).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img


def default_celeba_transform(image_size: int = 64):
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),  # to [-1, 1]
        ]
    )


def create_dataloaders(
    data_path: str,
    batch_size: int = 128,
    train_ratio: float = 0.9,
    num_workers: int = 8,
    pin_memory: bool = True,
    image_size: int = 64,
    seed: int = 42,
    use_zip: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    transform = default_celeba_transform(image_size=image_size)

    if use_zip:
        dataset = CelebAZipDataset(
            zip_path=data_path,
            transform=transform,
        )
    else:
        dataset = CelebAFolderDataset(
            root_dir=data_path,
            transform=transform,
        )

    total_len = len(dataset)
    train_len = int(total_len * train_ratio)
    val_len = total_len - train_len

    if train_len <= 0 or val_len <= 0:
        raise ValueError(
            f"Invalid split: total={total_len}, train_ratio={train_ratio}, "
            f"train={train_len}, val={val_len}"
        )

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        dataset, [train_len, val_len], generator=generator
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    return train_loader, val_loader


# ============================================================
# COMMON BUILDING BLOCKS
# ============================================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=4, s=2, p=1, use_bn=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, k, s, p, bias=not use_bn)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DeconvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=4, s=2, p=1, use_bn=True, final=False):
        super().__init__()
        layers = [nn.ConvTranspose2d(in_ch, out_ch, k, s, p, bias=not use_bn)]
        if not final:
            if use_bn:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
        else:
            layers.append(nn.Tanh())
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


# ============================================================
# VAE
# ============================================================

class VAEEncoder(nn.Module):
    def __init__(self, latent_dim: int = 128, in_channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, 64, use_bn=False),   # 64 -> 32
            ConvBlock(64, 128),                         # 32 -> 16
            ConvBlock(128, 256),                        # 16 -> 8
            ConvBlock(256, 512),                        # 8 -> 4
        )
        self.fc_mu = nn.Linear(512 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(512 * 4 * 4, latent_dim)

    def forward(self, x):
        h = self.features(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class VAEDecoder(nn.Module):
    def __init__(self, latent_dim: int = 128, out_channels: int = 3):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 512 * 4 * 4)
        self.deconv = nn.Sequential(
            DeconvBlock(512, 256),                      # 4 -> 8
            DeconvBlock(256, 128),                      # 8 -> 16
            DeconvBlock(128, 64),                       # 16 -> 32
            DeconvBlock(64, out_channels, final=True),  # 32 -> 64
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(z.size(0), 512, 4, 4)
        return self.deconv(h)


class VAE(nn.Module):
    def __init__(self, latent_dim: int = 128, in_channels: int = 3):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = VAEEncoder(latent_dim=latent_dim, in_channels=in_channels)
        self.decoder = VAEDecoder(latent_dim=latent_dim, out_channels=in_channels)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    @torch.no_grad()
    def sample(self, n: int, device: torch.device):
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decoder(z)


# ============================================================
# GAN (DCGAN)
# ============================================================

class GANGenerator(nn.Module):
    def __init__(self, latent_dim: int = 128, out_channels: int = 3):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),  # 1 -> 4
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),         # 4 -> 8
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),         # 8 -> 16
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),          # 16 -> 32
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, out_channels, 4, 2, 1, bias=False), # 32 -> 64
            nn.Tanh(),
        )

    def forward(self, z):
        if z.dim() == 2:
            z = z.unsqueeze(-1).unsqueeze(-1)
        return self.net(z)


class GANDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1, bias=False),   # 64 -> 32
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),           # 32 -> 16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),          # 16 -> 8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),          # 8 -> 4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, 1, 0, bias=False),            # 4 -> 1
        )

    def forward(self, x):
        return self.net(x).view(-1)


class GAN(nn.Module):
    def __init__(self, latent_dim: int = 128, channels: int = 3):
        super().__init__()
        self.latent_dim = latent_dim
        self.generator = GANGenerator(latent_dim=latent_dim, out_channels=channels)
        self.discriminator = GANDiscriminator(in_channels=channels)

    @torch.no_grad()
    def sample(self, n: int, device: torch.device):
        z = torch.randn(n, self.latent_dim, device=device)
        return self.generator(z)


# ============================================================
# DIFFUSION MODEL (DDPM)
# ============================================================

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor):

        half_dim = self.dim // 2
        emb_scale = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb_scale)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.norm1 = nn.BatchNorm2d(out_ch)
        self.norm2 = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU()
        self.res_conv = (
            nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)

        time_term = self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + time_term

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)

        return h + self.res_conv(x)


class SimpleUNet(nn.Module):
    def __init__(self, in_channels: int = 3, base_ch: int = 64, time_dim: int = 256):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.in_conv = nn.Conv2d(in_channels, base_ch, 3, padding=1)

        self.down1 = ResBlock(base_ch, base_ch, time_dim)
        self.pool1 = nn.MaxPool2d(2)  # 64 -> 32
        self.down2 = ResBlock(base_ch, base_ch * 2, time_dim)
        self.pool2 = nn.MaxPool2d(2)  # 32 -> 16

        self.mid = ResBlock(base_ch * 2, base_ch * 4, time_dim)

        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")  # 16 -> 32
        self.up_block1 = ResBlock(base_ch * 4 + base_ch * 2, base_ch * 2, time_dim)

        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")  # 32 -> 64
        self.up_block2 = ResBlock(base_ch * 2 + base_ch, base_ch, time_dim)

        self.out_conv = nn.Conv2d(base_ch, in_channels, 1)

    def forward(self, x, t):
        t_emb = self.time_embed(t)

        x0 = self.in_conv(x)
        x1 = self.down1(x0, t_emb)
        x2 = self.pool1(x1)

        x3 = self.down2(x2, t_emb)
        x4 = self.pool2(x3)

        xm = self.mid(x4, t_emb)

        xu = self.up1(xm)
        xu = torch.cat([xu, x3], dim=1)
        xu = self.up_block1(xu, t_emb)

        xu = self.up2(xu)
        xu = torch.cat([xu, x1], dim=1)
        xu = self.up_block2(xu, t_emb)

        return self.out_conv(xu)


class DiffusionModel(nn.Module):

    def __init__(
        self,
        image_channels: int = 3,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ):
        super().__init__()
        self.timesteps = timesteps
        self.denoiser = SimpleUNet(in_channels=image_channels)

        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alpha_hat = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_hat", alpha_hat)
        self.register_buffer("sqrt_alpha_hat", torch.sqrt(alpha_hat))
        self.register_buffer("sqrt_one_minus_alpha_hat", torch.sqrt(1.0 - alpha_hat))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None):
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alpha_hat_t = self.sqrt_alpha_hat[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_hat_t = self.sqrt_one_minus_alpha_hat[t].view(-1, 1, 1, 1)

        xt = sqrt_alpha_hat_t * x0 + sqrt_one_minus_alpha_hat_t * noise
        return xt, noise

    def forward(self, x0: torch.Tensor, t: torch.Tensor):
        xt, noise = self.q_sample(x0, t)
        pred_noise = self.denoiser(xt, t)
        return pred_noise, noise

    @torch.no_grad()
    def sample(self, n: int, device: torch.device, img_size: int = 64):
        x = torch.randn(n, 3, img_size, img_size, device=device)

        for step in reversed(range(self.timesteps)):
            t = torch.full((n,), step, device=device, dtype=torch.long)

            beta_t = self.betas[t].view(-1, 1, 1, 1)
            alpha_t = self.alphas[t].view(-1, 1, 1, 1)
            alpha_hat_t = self.alpha_hat[t].view(-1, 1, 1, 1)
            pred_noise = self.denoiser(x, t)

            if step > 0:
                z = torch.randn_like(x)
            else:
                z = torch.zeros_like(x)

            x = (
                (1.0 / torch.sqrt(alpha_t))
                * (x - ((1 - alpha_t) / torch.sqrt(1 - alpha_hat_t)) * pred_noise)
                + torch.sqrt(beta_t) * z
            )

        x = torch.clamp(x, -1.0, 1.0)
        return x


# ============================================================
# TRAINER CONFIGURATION
# ============================================================

@dataclass
class TrainerConfig:
    model_type: str
    epochs: int = 10
    lr: float = 2e-4
    batch_size: int = 128
    latent_dim: int = 128
    onnx_every: int = 5
    out_dir: str = "outputs/genmodels"
    device: str = "cuda"
    train_ratio: float = 0.9
    diffusion_steps: int = 200
    grad_clip: Optional[float] = None


# ============================================================
# GENERIC TRAINER
# ============================================================

class GenModelTrainer:

    def __init__(self, model: nn.Module, config: TrainerConfig):
        self.model = model
        self.config = config
        self.model_type = config.model_type.lower()
        self.device = torch.device(
            config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu"
        )
        self.model.to(self.device)

        os.makedirs(self.config.out_dir, exist_ok=True)

        if self.model_type == "vae":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        elif self.model_type == "gan":
            self.opt_g = torch.optim.Adam(
                self.model.generator.parameters(), lr=self.config.lr, betas=(0.5, 0.999)
            )
            self.opt_d = torch.optim.Adam(
                self.model.discriminator.parameters(), lr=self.config.lr, betas=(0.5, 0.999)
            )
            self.bce = nn.BCEWithLogitsLoss()
        elif self.model_type == "diffusion":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

        self.use_amp = self.device.type == "cuda"
        amp_device = "cuda" if self.use_amp else "cpu"
        self.scaler = torch.amp.GradScaler(amp_device, enabled=self.use_amp)

    def _vae_loss(self, recon_x, x, mu, logvar):
        recon_loss = F.mse_loss(recon_x, x, reduction="mean")
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total = recon_loss + 1e-3 * kl
        return total, recon_loss, kl

    def _train_step_vae(self, x):
        self.optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=self.use_amp):
            recon, mu, logvar = self.model(x)
            loss, recon_loss, kl = self._vae_loss(recon, x, mu, logvar)

        self.scaler.scale(loss).backward()
        if self.config.grad_clip is not None:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return {
            "loss": float(loss.item()),
            "recon_loss": float(recon_loss.item()),
            "kl": float(kl.item()),
        }

    def _train_step_gan(self, x):
        bsz = x.size(0)
        real = x
        real_labels = torch.ones(bsz, device=self.device)
        fake_labels = torch.zeros(bsz, device=self.device)

        # Train D
        self.opt_d.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=self.use_amp):
            real_logits = self.model.discriminator(real)
            d_real_loss = self.bce(real_logits, real_labels)

            z = torch.randn(bsz, self.model.latent_dim, device=self.device)
            fake = self.model.generator(z)
            fake_logits = self.model.discriminator(fake.detach())
            d_fake_loss = self.bce(fake_logits, fake_labels)

            d_loss = 0.5 * (d_real_loss + d_fake_loss)

        self.scaler.scale(d_loss).backward()
        self.scaler.step(self.opt_d)
        self.scaler.update()

        # Train G
        self.opt_g.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=self.use_amp):
            z = torch.randn(bsz, self.model.latent_dim, device=self.device)
            fake = self.model.generator(z)
            fake_logits = self.model.discriminator(fake)
            g_loss = self.bce(fake_logits, real_labels)

        self.scaler.scale(g_loss).backward()
        self.scaler.step(self.opt_g)
        self.scaler.update()

        return {
            "loss": float((d_loss + g_loss).item()),
            "d_loss": float(d_loss.item()),
            "g_loss": float(g_loss.item()),
        }

    def _train_step_diffusion(self, x):
        self.optimizer.zero_grad(set_to_none=True)
        bsz = x.size(0)
        t = torch.randint(0, self.model.timesteps, (bsz,), device=self.device).long()

        with torch.amp.autocast("cuda", enabled=self.use_amp):
            pred_noise, true_noise = self.model(x, t)
            loss = F.mse_loss(pred_noise, true_noise)

        self.scaler.scale(loss).backward()
        if self.config.grad_clip is not None:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return {"loss": float(loss.item())}

    def train_one_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.train()
        stats_sum: Dict[str, float] = {}
        steps = 0

        for batch in train_loader:
            x = batch.to(self.device, non_blocking=True)

            if self.model_type == "vae":
                stats = self._train_step_vae(x)
            elif self.model_type == "gan":
                stats = self._train_step_gan(x)
            elif self.model_type == "diffusion":
                stats = self._train_step_diffusion(x)
            else:
                raise RuntimeError("Unknown model type")

            for k, v in stats.items():
                stats_sum[k] = stats_sum.get(k, 0.0) + v
            steps += 1

        return {k: v / max(steps, 1) for k, v in stats_sum.items()}

    @torch.no_grad()
    def validate_one_epoch(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.eval()
        stats_sum: Dict[str, float] = {}
        steps = 0

        for batch in val_loader:
            x = batch.to(self.device, non_blocking=True)

            if self.model_type == "vae":
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    recon, mu, logvar = self.model(x)
                    loss, recon_loss, kl = self._vae_loss(recon, x, mu, logvar)
                stats = {
                    "val_loss": float(loss.item()),
                    "val_recon_loss": float(recon_loss.item()),
                    "val_kl": float(kl.item()),
                }

            elif self.model_type == "gan":
                bsz = x.size(0)
                real_labels = torch.ones(bsz, device=self.device)
                fake_labels = torch.zeros(bsz, device=self.device)

                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    real_logits = self.model.discriminator(x)
                    d_real_loss = self.bce(real_logits, real_labels)

                    z = torch.randn(bsz, self.model.latent_dim, device=self.device)
                    fake = self.model.generator(z)
                    fake_logits = self.model.discriminator(fake)
                    d_fake_loss = self.bce(fake_logits, fake_labels)

                    z2 = torch.randn(bsz, self.model.latent_dim, device=self.device)
                    fake2 = self.model.generator(z2)
                    fake_logits2 = self.model.discriminator(fake2)
                    g_loss = self.bce(fake_logits2, real_labels)

                stats = {
                    "val_d_loss": float((0.5 * (d_real_loss + d_fake_loss)).item()),
                    "val_g_loss": float(g_loss.item()),
                }

            elif self.model_type == "diffusion":
                bsz = x.size(0)
                t = torch.randint(0, self.model.timesteps, (bsz,), device=self.device).long()
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    pred_noise, true_noise = self.model(x, t)
                    loss = F.mse_loss(pred_noise, true_noise)
                stats = {"val_loss": float(loss.item())}

            else:
                raise RuntimeError("Unknown model type")

            for k, v in stats.items():
                stats_sum[k] = stats_sum.get(k, 0.0) + v
            steps += 1

        return {k: v / max(steps, 1) for k, v in stats_sum.items()}

    def save_checkpoint(self, epoch: int):
        ts = get_timestamp()
        ckpt_path = os.path.join(
            self.config.out_dir,
            f"{self.model_type}_epoch_{epoch}_{ts}.pt"
        )
        torch.save(
            {
                "epoch": epoch,
                "timestamp": ts,
                "model_type": self.model_type,
                "model_state_dict": self.model.state_dict(),
                "config": self.config.__dict__,
            },
            ckpt_path,
        )
        return ckpt_path

    def export_onnx(self, epoch: Optional[int] = None):
        ts = get_timestamp()
        suffix = f"_epoch_{epoch}" if epoch is not None else "_final"

        if self.model_type == "vae":
            export_model = self.model.decoder.eval().to(self.device)
            dummy_input = torch.randn(1, self.model.latent_dim, device=self.device)
            onnx_path = os.path.join(
                self.config.out_dir,
                f"vae_decoder{suffix}_{ts}.onnx"
            )
            torch.onnx.export(
                export_model,
                dummy_input,
                onnx_path,
                input_names=["z"],
                output_names=["image"],
                dynamic_axes={"z": {0: "batch"}, "image": {0: "batch"}},
                opset_version=18,
            )

        elif self.model_type == "gan":
            export_model = self.model.generator.eval().to(self.device)
            dummy_input = torch.randn(1, self.model.latent_dim, device=self.device)
            onnx_path = os.path.join(
                self.config.out_dir,
                f"gan_generator{suffix}_{ts}.onnx"
            )
            torch.onnx.export(
                export_model,
                dummy_input,
                onnx_path,
                input_names=["z"],
                output_names=["image"],
                dynamic_axes={"z": {0: "batch"}, "image": {0: "batch"}},
                opset_version=18,
            )

        elif self.model_type == "diffusion":
            export_model = self.model.denoiser.eval().to(self.device)
            dummy_x = torch.randn(1, 3, 64, 64, device=self.device)
            dummy_t = torch.zeros(1, dtype=torch.long, device=self.device)
            onnx_path = os.path.join(
                self.config.out_dir,
                f"diffusion_denoiser{suffix}_{ts}.onnx"
            )
            torch.onnx.export(
                export_model,
                (dummy_x, dummy_t),
                onnx_path,
                input_names=["x_t", "t"],
                output_names=["pred_noise"],
                dynamic_axes={
                    "x_t": {0: "batch"},
                    "t": {0: "batch"},
                    "pred_noise": {0: "batch"},
                },
                opset_version=18,
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        return onnx_path

    def fit(self, train_loader: DataLoader, val_loader: DataLoader):
        history = []

        for epoch in range(1, self.config.epochs + 1):
            train_stats = self.train_one_epoch(train_loader, epoch)
            val_stats = self.validate_one_epoch(val_loader, epoch)

            record = {"epoch": epoch, **train_stats, **val_stats}
            history.append(record)

            print(f"[{self.model_type.upper()}] Epoch {epoch}/{self.config.epochs} | {record}")

            ckpt_path = self.save_checkpoint(epoch)
            print(f"Saved checkpoint: {ckpt_path}")

            if self.config.onnx_every > 0 and epoch % self.config.onnx_every == 0:
                onnx_path = self.export_onnx(epoch=epoch)
                print(f"Exported ONNX: {onnx_path}")

        final_onnx = self.export_onnx(epoch=None)
        print(f"Final ONNX exported to: {final_onnx}")
        return history


# ============================================================
# FACTORY HELPERS
# ============================================================

def build_model(
    model_type: str,
    latent_dim: int = 128,
    diffusion_steps: int = 200,
    channels: int = 3,
) -> nn.Module:
    model_type = model_type.lower()
    if model_type == "vae":
        return VAE(latent_dim=latent_dim, in_channels=channels)
    if model_type == "gan":
        return GAN(latent_dim=latent_dim, channels=channels)
    if model_type == "diffusion":
        return DiffusionModel(image_channels=channels, timesteps=diffusion_steps)
    raise ValueError(f"Unsupported model_type: {model_type}")

