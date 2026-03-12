import argparse
import subprocess
import torch
import matplotlib.pyplot as plt

from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms

from deepl.multiclass import ImageNetCNN, CNNTrainer


# ------------------------------------------------------------
# GPU SELECTION (Least utilized GPU)
# ------------------------------------------------------------
def get_best_gpu(strategy="utilization"):

    if strategy == "utilization":

        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
        )

        utilizations = [
            int(x.strip()) for x in result.stdout.strip().split("\n")
        ]

        return utilizations.index(min(utilizations))

    else:

        free_mem = []

        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            free_mem.append(free)

        return free_mem.index(max(free_mem))


# ------------------------------------------------------------
# COLLATE FUNCTION
# ------------------------------------------------------------
def collate_fn(batch):

    pixel_values = torch.stack(
        [item["pixel_values"] for item in batch]
    )

    labels = torch.tensor(
        [item["labels"] for item in batch]
    )

    return {
        "pixel_values": pixel_values,
        "labels": labels,
    }


# ------------------------------------------------------------
# MAIN SCRIPT
# ------------------------------------------------------------
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train_ratio", type=float, default=0.10)
    parser.add_argument("--val_ratio", type=float, default=0.05)

    args = parser.parse_args()

    # --------------------------------------------------------
    # Load ImageNet Dataset
    # --------------------------------------------------------
    dataset = load_dataset(
        "ILSVRC/imagenet-1k",
        cache_dir="/data/CPE_487-587/imagenet-1k",
    )

    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]

    class_names = train_dataset.features["label"].names
    num_classes = len(class_names)

    print("Number of classes:", num_classes)

    # --------------------------------------------------------
    # Save example images
    # --------------------------------------------------------
    train_example = train_dataset[0]
    val_example = val_dataset[0]

    plt.figure(figsize=(6, 6))
    plt.imshow(train_example["image"])
    plt.axis("off")
    plt.title("Example Training Image")
    plt.savefig("example_train_image.png")
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.imshow(val_example["image"])
    plt.axis("off")
    plt.title("Example Validation Image")
    plt.savefig("example_val_image.png")
    plt.close()

    # --------------------------------------------------------
    # Subset selection
    # --------------------------------------------------------
    train_size = int(len(train_dataset) * args.train_ratio)
    val_size = int(len(val_dataset) * args.val_ratio)

    train_dataset = train_dataset.select(range(train_size))
    val_dataset = val_dataset.select(range(val_size))

    # --------------------------------------------------------
    # Image transforms
    # --------------------------------------------------------
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        ),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        ),
    ])

    # --------------------------------------------------------
    # Dataset preprocessing
    # --------------------------------------------------------
    def preprocess_train(examples):

        images = [
            train_transform(img.convert("RGB"))
            for img in examples["image"]
        ]

        return {
            "pixel_values": images,
            "labels": examples["label"],
        }

    def preprocess_val(examples):

        images = [
            val_transform(img.convert("RGB"))
            for img in examples["image"]
        ]

        return {
            "pixel_values": images,
            "labels": examples["label"],
        }

    train_dataset = train_dataset.with_transform(preprocess_train)
    val_dataset = val_dataset.with_transform(preprocess_val)

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # --------------------------------------------------------
    # Select best GPU
    # --------------------------------------------------------
    device_id = get_best_gpu(strategy="utilization")

    device = torch.device(f"cuda:{device_id}")

    print("Using GPU:", device_id)

    # --------------------------------------------------------
    # Instantiate model and trainer
    # --------------------------------------------------------
    model = ImageNetCNN(num_classes=num_classes)

    trainer = CNNTrainer(model, device=device)

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------
    losses, train_acc, val_acc = trainer.train(
        train_loader,
        val_loader,
        args.epochs,
    )

    # --------------------------------------------------------
    # Export ONNX model
    # --------------------------------------------------------
    trainer.export_onnx("imagenet_model.onnx")

    # --------------------------------------------------------
    # Plot loss and accuracy
    # --------------------------------------------------------
    plt.figure()

    plt.plot(losses, label="Loss")
    plt.plot(train_acc, label="Train Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.legend()

    plt.title("Training Progress")

    plt.savefig("training_metrics.png")

    print("Training plot saved as training_metrics.png")


if __name__ == "__main__":
    main()
