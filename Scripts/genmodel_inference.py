import argparse
import json
import os
import re
from datetime import datetime

import numpy as np
import onnxruntime as ort
import torch

from mypythonpackage.deepl.gen_model import build_model
from mypythonpackage.metrics.image_quality import (
    aggregate_metric_dict,
    compute_all_metrics,
    save_image_grid,
    save_metric_barplot,
)


def get_timestamp() -> str:

    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def find_latest_pt(model_dir: str, prefix: str) -> str:

    files = [
        f for f in os.listdir(model_dir)
        if f.startswith(prefix) and f.endswith(".pt")
    ]
    if not files:
        raise FileNotFoundError(
            f"No checkpoint files found in {model_dir} with prefix {prefix}"
        )

    def extract_epoch(name: str) -> int:
        match = re.search(r"_epoch_(\d+)", name)
        if match:
            return int(match.group(1))
        return -1

    files.sort(key=extract_epoch)
    return os.path.join(model_dir, files[-1])


def load_pytorch_model(
    model_type: str,
    model_dir: str,
    latent_dim: int,
    diffusion_steps: int,
    device: str,
):
    model = build_model(
        model_type=model_type,
        latent_dim=latent_dim,
        diffusion_steps=diffusion_steps,
        channels=3,
    )

    ckpt_path = find_latest_pt(model_dir, f"{model_type}_epoch_")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"Loaded PyTorch checkpoint: {ckpt_path}")
    return model


@torch.no_grad()
def sample_with_pytorch(model_type: str, model, n: int, device: str):
    device_obj = torch.device(device)

    if model_type == "vae":
        return model.sample(n, device=device_obj)
    if model_type == "gan":
        return model.sample(n, device=device_obj)
    if model_type == "diffusion":
        return model.sample(n, device=device_obj, img_size=64)

    raise ValueError(f"Unsupported model type: {model_type}")


def run_onnx_generator(onnx_path: str, z: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    outputs = session.run(None, {"z": z.astype(np.float32)})
    return outputs[0]


def run_onnx_diffusion_sampling(
    onnx_path: str,
    n: int,
    timesteps: int,
    image_size: int = 64,
) -> np.ndarray:

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    betas = np.linspace(1e-4, 2e-2, timesteps, dtype=np.float32)
    alphas = 1.0 - betas
    alpha_hat = np.cumprod(alphas, axis=0)

    x = np.random.randn(n, 3, image_size, image_size).astype(np.float32)

    for step in reversed(range(timesteps)):
        t = np.full((n,), step, dtype=np.int64)
        pred_noise = session.run(None, {"x_t": x, "t": t})[0]

        beta_t = betas[step]
        alpha_t = alphas[step]
        alpha_hat_t = alpha_hat[step]

        if step > 0:
            z = np.random.randn(*x.shape).astype(np.float32)
        else:
            z = np.zeros_like(x, dtype=np.float32)

        x = (
            (1.0 / np.sqrt(alpha_t))
            * (x - ((1.0 - alpha_t) / np.sqrt(1.0 - alpha_hat_t)) * pred_noise)
            + np.sqrt(beta_t) * z
        ).astype(np.float32)

    x = np.clip(x, -1.0, 1.0)
    return x


def infer_single_model(args, model_name: str, summary_results: dict, run_timestamp: str):
    print(f"\n===== Inference for {model_name.upper()} =====")

    model_dir = os.path.join(args.model_dir, model_name)
    os.makedirs(args.out_dir, exist_ok=True)

    model = load_pytorch_model(
        model_type=model_name,
        model_dir=model_dir,
        latent_dim=args.latent_dim,
        diffusion_steps=args.diffusion_steps,
        device=args.device,
    )

    samples = sample_with_pytorch(
        model_type=model_name,
        model=model,
        n=args.num_samples,
        device=args.device,
    )

    samples_cpu = samples.detach().cpu()

    grid_path = os.path.join(
        args.out_dir,
        f"{model_name}_samples_{args.num_samples}_{run_timestamp}.png",
    )
    save_image_grid(samples_cpu, grid_path, nrow=5)
    print(f"Saved sample grid: {grid_path}")

    metric_tensors = compute_all_metrics(samples_cpu)
    metric_means = aggregate_metric_dict(metric_tensors)

    metrics_path = os.path.join(
        args.out_dir,
        f"{model_name}_metrics_{run_timestamp}.json",
    )
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metric_means, f, indent=2)
    print(f"Saved metrics: {metrics_path}")

    summary_results[model_name] = metric_means


def main():
    parser = argparse.ArgumentParser(
        description="Inference and evaluation for generative models"
    )

    parser.add_argument("--model-dir", type=str, default="outputs/genmodels")
    parser.add_argument("--out-dir", type=str, default="outputs/gen_eval")
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--diffusion-steps", type=int, default=200)
    parser.add_argument("--num-samples", type=int, default=25)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    args.model_dir = os.path.expanduser(args.model_dir)
    args.out_dir = os.path.expanduser(args.out_dir)

    os.makedirs(args.out_dir, exist_ok=True)

    run_timestamp = get_timestamp()
    summary_results = {}

    for model_name in ["vae", "gan", "diffusion"]:
        infer_single_model(args, model_name, summary_results, run_timestamp)

    summary_json = os.path.join(
        args.out_dir,
        f"comparison_metrics_{run_timestamp}.json",
    )
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2)

    plot_path = os.path.join(
        args.out_dir,
        f"comparison_barplot_{run_timestamp}.png",
    )
    save_metric_barplot(summary_results, plot_path)

    print(f"\nSaved comparison metrics: {summary_json}")
    print(f"Saved comparison barplot: {plot_path}")


if __name__ == "__main__":
    main()

