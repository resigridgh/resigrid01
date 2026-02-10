import argparse
import glob
import os
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description="Aggregate HW02Q8 metrics and produce boxplots")
    p.add_argument("--keyword", type=str, required=True, help="keyword used in metrics filenames")
    p.add_argument("--metrics_dir", type=str, default="outputs/metrics")
    p.add_argument("--out_dir", type=str, default="outputs/plots")
    return p.parse_args()


def main():
    args = parse_args()

    pattern = os.path.join(args.metrics_dir, f"metrics_{args.keyword}_*.csv")
    files = sorted(glob.glob(pattern))

    if len(files) == 0:
        raise FileNotFoundError(f"No metrics files found matching: {pattern}")

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    os.makedirs(args.out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # For each metric, boxplot train vs test
    metrics = ["accuracy", "f1", "precision", "recall"]
    for m in metrics:
        train_col = f"train_{m}"
        test_col = f"test_{m}"
        if train_col not in df.columns or test_col not in df.columns:
            continue

        plt.figure()
        plt.boxplot([df[train_col].values, df[test_col].values], labels=["Train", "Test"])
        plt.ylabel(m.upper())
        plt.title(f"{m.upper()} Boxplot ({args.keyword})")

        out_path = os.path.join(args.out_dir, f"boxplot_{m}_{args.keyword}_{ts}.png")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_path}")

    # Also save the aggregated table
    out_csv = os.path.join(args.out_dir, f"aggregate_{args.keyword}_{ts}.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved aggregate CSV: {out_csv}")


if __name__ == "__main__":
    main()
