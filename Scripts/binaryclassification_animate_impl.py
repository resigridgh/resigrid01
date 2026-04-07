import argparse
import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from mypythonpackage.deepl.two_layer_binary_classification import binary_classification
from mypythonpackage.animation import animate_weight_heatmap, animate_large_heatmap


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate weight-matrix animations for binary classification"
    )

    parser.add_argument("--dt", type=float, default=0.04, help="Time step between frames")
    parser.add_argument("--epochs", type=int, default=5000, help="Number of training epochs")
    parser.add_argument("--eta", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--num_features", type=int, default=200, help="Number of input features")
    parser.add_argument("--num_samples", type=int, default=40000, help="Number of samples")
    parser.add_argument(
        "--media_dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "media"),
        help="Directory to save mp4 animations",
    )
    parser.add_argument(
        "--large_threshold",
        type=int,
        default=300,
        help="Use large heatmap animator if max(rows, cols) >= threshold",
    )

    return parser.parse_args()


def choose_animator(weight_stack: torch.Tensor, threshold: int):
    _, rows, cols = weight_stack.shape

    if max(rows, cols) >= threshold:
        return animate_large_heatmap
    return animate_weight_heatmap


def main():
    args = parse_args()

    os.makedirs(args.media_dir, exist_ok=True)

    print("Starting binary classification training...")
    print(f"Features : {args.num_features}")
    print(f"Samples  : {args.num_samples}")
    print(f"Epochs   : {args.epochs}")
    print(f"Eta      : {args.eta}")
    print(f"dt       : {args.dt}")

    results = binary_classification(
        d=args.num_features,
        n=args.num_samples,
        epochs=args.epochs,
        eta=args.eta,
    )

    train_losses = results[0]
    weights_W1 = results[1]
    weights_W2 = results[2]
    weights_W3 = results[3]
    weights_W4 = results[4]

    weight_dict = {
        "W1": weights_W1.detach().cpu(),
        "W2": weights_W2.detach().cpu(),
        "W3": weights_W3.detach().cpu(),
        "W4": weights_W4.detach().cpu(),
    }

    print("Training finished.")
    print(f"Final loss: {train_losses[-1].item():.6f}")

    for name, stack in weight_dict.items():
        animator = choose_animator(stack, args.large_threshold)

        file_name = os.path.join(args.media_dir, f"{name.lower()}_animation")
        title_str = f"{name} Weight Evolution"

        print(f"Creating animation for {name} with shape {tuple(stack.shape)}")

        animator(
            stack,
            dt=args.dt,
            file_name=file_name,
            title_str=title_str,
        )

    print(f"All animations saved in: {args.media_dir}")


if __name__ == "__main__":
    main()
