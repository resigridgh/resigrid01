#!/usr/bin/env python3
import argparse
import json
import os
import random

import numpy as np
import torch

from mypythonpackage.deepl.gen_model import (
    GenModelTrainer,
    TrainerConfig,
    build_model,
    create_dataloaders,
)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True


def train_single_model(args, model_name: str):
    print(f"\n===== Training {model_name.upper()} =====")
    model_out_dir = os.path.join(args.out_dir, model_name)
    os.makedirs(model_out_dir, exist_ok=True)

    train_loader, val_loader = create_dataloaders(
        data_path=os.path.expanduser(args.data_path),
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        num_workers=args.num_workers,
        pin_memory=True,
        image_size=args.image_size,
        seed=args.seed,
        use_zip=args.use_zip,
    )

    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    model = build_model(
        model_type=model_name,
        latent_dim=args.latent_dim,
        diffusion_steps=args.diffusion_steps,
        channels=3,
    )

    config = TrainerConfig(
        model_type=model_name,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        onnx_every=args.onnx_every,
        out_dir=model_out_dir,
        device=args.device,
        train_ratio=args.train_ratio,
        diffusion_steps=args.diffusion_steps,
        grad_clip=args.grad_clip,
    )

    trainer = GenModelTrainer(model=model, config=config)
    history = trainer.fit(train_loader, val_loader)

    hist_path = os.path.join(model_out_dir, f"{model_name}_history.json")
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Saved history to: {hist_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train VAE/GAN/Diffusion on CelebA dataset"
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default="~/datasets/img_align_celeba/img_align_celeba",
        help="Path to extracted CelebA folder or zip file",
    )
    parser.add_argument(
        "--use-zip",
        action="store_true",
        help="Use zip loader instead of folder loader",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["vae", "gan", "diffusion", "all"],
        help="Which model to train",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--train-ratio", type=float, default=0.09)
    parser.add_argument("--onnx-every", type=int, default=5)
    parser.add_argument("--diffusion-steps", type=int, default=200)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out-dir", type=str, default="outputs/genmodels")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad-clip", type=float, default=None)

    args = parser.parse_args()
    set_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    expanded_path = os.path.expanduser(args.data_path)
    if not os.path.exists(expanded_path):
        raise FileNotFoundError(f"Data path does not exist: {expanded_path}")

    print(f"Using data path: {expanded_path}")
    print(f"Using zip loader: {args.use_zip}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Requested device: {args.device}")

    model_list = ["vae", "gan", "diffusion"] if args.model == "all" else [args.model]

    for model_name in model_list:
        train_single_model(args, model_name)

    print("\nAll requested training jobs finished.")


if __name__ == "__main__":
    main()
