import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


# -------------------------------------------------------------------
# Make sure src/ is on Python path when running:
#   python Scripts/multiclass_impl.py
# -------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

# Direct import from module to avoid __init__.py export issues
from mypythonpackage.deepl.multiclass import SimpleNN, ClassTrainer


DROP_COLS = [
    "Flow ID",
    "Source IP",
    "Source Port",
    "Destination IP",
    "Destination Port",
    "Protocol",
    "Timestamp",
]


def parse_args():
    p = argparse.ArgumentParser(description="HW02Q8 Multi-class NN training")
    p.add_argument(
        "--data_path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data", "Android_Malware.csv"),
    )
    p.add_argument("--eta", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"])
    p.add_argument("--device", type=str, default=None, help="cuda or cpu (default: auto)")
    p.add_argument("--keyword", type=str, default="hw02", help="unique keyword for file naming")
    p.add_argument("--save_onnx", action="store_true")
    p.add_argument("--onnx_name", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(
            f"Dataset not found at {args.data_path}. "
            f"Place Android_Malware.csv inside {os.path.join(PROJECT_ROOT, 'data')}"
        )

    # low_memory=False removes the mixed-type warning
    df = pd.read_csv(args.data_path, low_memory=False)

    # Identify label column
    possible_label_cols = ["Label", "label", "Class", "class", "Malware", "Category"]
    label_col = None
    for c in possible_label_cols:
        if c in df.columns:
            label_col = c
            break

    if label_col is None:
        label_col = df.columns[-1]

    # Drop known non-useful columns if present
    drop_now = [c for c in DROP_COLS if c in df.columns]
    if drop_now:
        df = df.drop(columns=drop_now)

    # Separate X and y
    y_raw = df[label_col].astype(str).values
    X_df = df.drop(columns=[label_col])

    # Convert all features to numeric
    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    X_df = X_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_names = list(le.classes_)
    num_classes = len(class_names)

    if num_classes < 2:
        raise ValueError("Need at least 2 classes for multi-class classification.")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_df.values,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Torch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    # Model
    model = SimpleNN(
        in_features=X_train_t.shape[1],
        num_classes=num_classes,
    )

    # Trainer
    trainer = ClassTrainer(
        X_train=X_train_t,
        y_train=y_train_t,
        eta=args.eta,
        epoch=args.epochs,
        loss=nn.CrossEntropyLoss(),
        optimizer_name=args.optimizer,
        model=model,
        device=args.device,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    # Train
    trainer.train()

    # Train predictions
    with torch.no_grad():
        train_pred = trainer.predict(X_train_t).cpu().numpy()
    train_true = y_train

    # Test metrics
    test_metrics = trainer.test(X_test_t, y_test_t)

    train_metrics = {
        "accuracy": float(accuracy_score(train_true, train_pred)),
        "f1": float(f1_score(train_true, train_pred, average="macro", zero_division=0)),
        "precision": float(precision_score(train_true, train_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(train_true, train_pred, average="macro", zero_division=0)),
    }

    # Output directories
    plots_dir = os.path.join(PROJECT_ROOT, "outputs", "plots")
    metrics_dir = os.path.join(PROJECT_ROOT, "outputs", "metrics")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    # Plots
    trainer.evaluation(
        class_names=class_names,
        out_dir=plots_dir,
        tag=args.keyword,
    )

    # Save ONNX if requested
    onnx_path = None
    if args.save_onnx:
        if args.onnx_name is not None:
            if os.path.isabs(args.onnx_name):
                onnx_path = trainer.save(args.onnx_name)
            else:
                onnx_path = trainer.save(os.path.join(PROJECT_ROOT, args.onnx_name))
        else:
            ts_onnx = datetime.now().strftime("%Y%m%d_%H%M%S")
            onnx_path = trainer.save(
                os.path.join(PROJECT_ROOT, "outputs", f"model_{args.keyword}_{ts_onnx}.onnx")
            )

    # Save metrics CSV
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(metrics_dir, f"metrics_{args.keyword}_{ts}.csv")

    row = {
        "timestamp": ts,
        "keyword": args.keyword,
        "data_path": args.data_path,
        "epochs": args.epochs,
        "eta": args.eta,
        "optimizer": args.optimizer,
        "batch_size": args.batch_size,
        "test_size": args.test_size,
        "num_classes": num_classes,
        "train_accuracy": train_metrics["accuracy"],
        "train_f1": train_metrics["f1"],
        "train_precision": train_metrics["precision"],
        "train_recall": train_metrics["recall"],
        "test_accuracy": test_metrics.accuracy,
        "test_f1": test_metrics.f1,
        "test_precision": test_metrics.precision,
        "test_recall": test_metrics.recall,
        "onnx_path": "" if onnx_path is None else onnx_path,
    }

    pd.DataFrame([row]).to_csv(out_csv, index=False)

    print(f"Training finished.")
    print(f"Train accuracy: {train_metrics['accuracy']:.4f}")
    print(f"Test  accuracy: {test_metrics.accuracy:.4f}")
    print(f"Saved metrics to: {out_csv}")
    if onnx_path is not None:
        print(f"Saved ONNX to: {onnx_path}")


if __name__ == "__main__":
    main()
