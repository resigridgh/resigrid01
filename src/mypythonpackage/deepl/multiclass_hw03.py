from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim


# =========================================================
# CNN BLOCK
# Conv → BatchNorm → ReLU → MaxPool
# =========================================================

class ConvLayer(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# =========================================================
# IMAGENET CNN MODEL (Figure-1 architecture)
# =========================================================

class ImageNetCNN(nn.Module):

    def __init__(self, num_classes: int = 1000):
        super().__init__()

        self.features = nn.Sequential(
            ConvLayer(3, 64),
            ConvLayer(64, 128),
            ConvLayer(128, 256),
            ConvLayer(256, 512),
            ConvLayer(512, 512),
        )

        # Global Average Pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)

        return x


# =========================================================
# CNN TRAINER (ImageNet)
# =========================================================

class CNNTrainer:

    def __init__(self, model: nn.Module, device=None, lr: float = 0.01):

        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)

        # Loss
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=1e-4
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=30,
            gamma=0.1
        )

    def accuracy(self, outputs: torch.Tensor, labels: torch.Tensor) -> float:

        preds = torch.argmax(outputs, dim=1)
        correct = (preds == labels).sum().item()

        return correct / labels.size(0)

    def train(self, train_loader, val_loader, epochs: int):

        train_losses = []
        train_acc = []
        val_acc = []

        for epoch in range(epochs):

            self.model.train()

            running_loss = 0
            running_acc = 0

            for i, batch in enumerate(train_loader):

                images = batch["pixel_values"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(images)

                loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                acc = self.accuracy(outputs, labels)

                running_loss += loss.item()
                running_acc += acc

                if i % 10 == 0:
                    print(
                        f"Epoch {epoch} Batch {i} "
                        f"Loss {loss.item():.4f} "
                        f"Acc {acc:.4f}"
                    )

            self.scheduler.step()

            train_losses.append(running_loss / len(train_loader))
            train_acc.append(running_acc / len(train_loader))

            val_accuracy = self.validate(val_loader)
            val_acc.append(val_accuracy)

            print("Validation Accuracy:", val_accuracy)

        return train_losses, train_acc, val_acc

    def validate(self, loader):

        self.model.eval()

        total = 0
        correct = 0

        with torch.no_grad():

            for batch in loader:

                images = batch["pixel_values"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(images)

                preds = torch.argmax(outputs, dim=1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

        return correct / total

    def export_onnx(self, path: str = "imagenet_model.onnx"):

        dummy = torch.randn(1, 3, 224, 224).to(self.device)

        torch.onnx.export(
            self.model,
            dummy,
            path,
            opset_version=17,
            input_names=["images"],
            output_names=["logits"],
            dynamic_axes={
                "images": {0: "batch"},
                "logits": {0: "batch"},
            },
        )

        print("ONNX model saved to", path)
