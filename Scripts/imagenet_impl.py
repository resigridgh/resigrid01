import argparse
import torch
import matplotlib.pyplot as plt

from datasets import load_dataset
from torch.utils.data import DataLoader

from deepl.multiclass import ImageNetCNN
from deepl.trainer import CNNTrainer


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train_ratio", type=float, default=0.1)
    parser.add_argument("--val_ratio", type=float, default=0.05)

    args = parser.parse_args()

    dataset = load_dataset(
        "ILSVRC/imagenet-1k",
        cache_dir="/data/CPE_487-587/imagenet-1k"
    )

    train_size = int(len(dataset['train']) * args.train_ratio)
    val_size = int(len(dataset['validation']) * args.val_ratio)

    train_dataset = dataset['train'].select(range(train_size))
    val_dataset = dataset['validation'].select(range(val_size))

    train_loader = DataLoader(train_dataset, batch_size=128)
    val_loader = DataLoader(val_dataset, batch_size=128)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ImageNetCNN(num_classes=1000)

    trainer = CNNTrainer(model, device)

    losses, train_acc, val_acc = trainer.train(
        train_loader,
        val_loader,
        args.epochs
    )

    trainer.export_onnx("imagenet_model.onnx")

    plt.plot(losses)
    plt.title("Training Loss")
    plt.savefig("training_plot.png")


if __name__ == "__main__":
    main()
