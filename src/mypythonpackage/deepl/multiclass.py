
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


class SimpleNN(nn.Module):
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()

        if in_features <= 0:
            raise ValueError("in_features must be > 0")
        if num_classes <= 1:
            raise ValueError("num_classes must be >= 2")

        self.fc1 = nn.Linear(in_features, 3)
        self.fc2 = nn.Linear(3, 4)
        self.fc3 = nn.Linear(4, 5)
        self.fc4 = nn.Linear(5, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x


@dataclass
class EvalMetrics:
    accuracy: float
    f1: float
    precision: float
    recall: float


class ClassTrainer:
    def __init__(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        eta: float = 1e-3,
        epoch: int = 200,
        loss: Optional[nn.Module] = None,
        optimizer_name: str = "adam",
        model: Optional[nn.Module] = None,
        device: Optional[str] = None,
        batch_size: int = 1024,
        seed: int = 42,
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.X_train = X_train.to(self.device)
        self.y_train = y_train.to(self.device)
        self.eta = eta
        self.epoch = epoch
        self.batch_size = batch_size

        if model is None:
            raise ValueError("model must be provided")

        self.model = model.to(self.device)
        self.loss = loss if loss is not None else nn.CrossEntropyLoss()

        if optimizer_name.lower() == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.eta)
        elif optimizer_name.lower() == "sgd":
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.eta, momentum=0.9)
        else:
            raise ValueError("optimizer_name must be either 'adam' or 'sgd'")

        self.loss_vector = torch.zeros(self.epoch, device=self.device)
        self.accuracy_vector = torch.zeros(self.epoch, device=self.device)

        self._train_true = None
        self._train_pred = None
        self._test_true = None
        self._test_pred = None

    def _make_batches(self, X: torch.Tensor, y: torch.Tensor):
        n = X.shape[0]
        idx = torch.randperm(n, device=X.device)
        for start in range(0, n, self.batch_size):
            batch_idx = idx[start:start + self.batch_size]
            yield X[batch_idx], y[batch_idx]

    def train(self):
        self.model.train()

        for ep in range(self.epoch):
            losses = []
            preds_all = []
            true_all = []

            for xb, yb in self._make_batches(self.X_train, self.y_train):
                self.optimizer.zero_grad()
                logits = self.model(xb)
                loss = self.loss(logits, yb.long())
                loss.backward()
                self.optimizer.step()

                losses.append(loss.detach())
                preds_all.append(torch.argmax(logits.detach(), dim=1))
                true_all.append(yb.detach())

            mean_loss = torch.stack(losses).mean()
            y_true = torch.cat(true_all).cpu().numpy()
            y_pred = torch.cat(preds_all).cpu().numpy()
            acc = accuracy_score(y_true, y_pred)

            self.loss_vector[ep] = mean_loss
            self.accuracy_vector[ep] = acc

            print(f"Epoch [{ep+1}/{self.epoch}] Loss: {mean_loss.item():.6f} Accuracy: {acc:.6f}")

        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.X_train)
            preds = torch.argmax(logits, dim=1)

        self._train_true = self.y_train.cpu().numpy()
        self._train_pred = preds.cpu().numpy()

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        X = X.to(self.device)
        with torch.no_grad():
            logits = self.model(X)
            preds = torch.argmax(logits, dim=1)
        return preds

    def test(self, X_test: torch.Tensor, y_test: torch.Tensor) -> EvalMetrics:
        self.model.eval()
        X_test = X_test.to(self.device)
        y_test = y_test.to(self.device)

        with torch.no_grad():
            logits = self.model(X_test)
            preds = torch.argmax(logits, dim=1)

        y_true = y_test.cpu().numpy()
        y_pred = preds.cpu().numpy()

        self._test_true = y_true
        self._test_pred = y_pred

        return EvalMetrics(
            accuracy=float(accuracy_score(y_true, y_pred)),
            f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            precision=float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            recall=float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        )

    def evaluation(self, class_names=None, out_dir: str = "outputs/plots", tag: str = "run"):
        os.makedirs(out_dir, exist_ok=True)

        loss_path = os.path.join(out_dir, f"loss_{tag}.png")
        acc_path = os.path.join(out_dir, f"accuracy_{tag}.png")
        train_cm_path = os.path.join(out_dir, f"train_confusion_{tag}.png")
        test_cm_path = os.path.join(out_dir, f"test_confusion_{tag}.png")

        plt.figure(figsize=(6, 4))
        plt.plot(self.loss_vector.detach().cpu().numpy())
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.tight_layout()
        plt.savefig(loss_path, dpi=300)
        plt.close()

        plt.figure(figsize=(6, 4))
        plt.plot(self.accuracy_vector.detach().cpu().numpy())
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Training Accuracy")
        plt.tight_layout()
        plt.savefig(acc_path, dpi=300)
        plt.close()

        if self._train_true is not None and self._train_pred is not None:
            cm = confusion_matrix(self._train_true, self._train_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
            fig, ax = plt.subplots(figsize=(8, 6))
            disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
            plt.tight_layout()
            plt.savefig(train_cm_path, dpi=300)
            plt.close(fig)

        if self._test_true is not None and self._test_pred is not None:
            cm = confusion_matrix(self._test_true, self._test_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
            fig, ax = plt.subplots(figsize=(8, 6))
            disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
            plt.tight_layout()
            plt.savefig(test_cm_path, dpi=300)
            plt.close(fig)

        print("Plots saved")

        return {
            "loss_plot": loss_path,
            "accuracy_plot": acc_path,
            "train_confusion": train_cm_path,
            "test_confusion": test_cm_path,
        }

    def save(self, path: str = "model.onnx") -> str:
        out_dir = os.path.dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        dummy = torch.randn(1, self.X_train.shape[1], device=self.device)

        self.model.eval()
        torch.onnx.export(
            self.model,
            dummy,
            path,
            opset_version=17,
            input_names=["features"],
            output_names=["logits"],
            dynamic_axes={
                "features": {0: "batch"},
                "logits": {0: "batch"},
            },
        )

        print(f"ONNX model saved to {path}")
        return path


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.block(x)


class ImageNetCNN(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        self.features = nn.Sequential(
            ConvLayer(3, 64),
            ConvLayer(64, 128),
            ConvLayer(128, 256),
            ConvLayer(256, 512),
            ConvLayer(512, 512),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


class CNNTrainer:
    def __init__(self, model, device=None, lr=0.01):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=1e-4,
        )
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)

    def accuracy(self, outputs, labels):
        preds = torch.argmax(outputs, dim=1)
        correct = (preds == labels).sum().item()
        return correct / labels.size(0)

    def train(self, train_loader, val_loader, epochs):
        train_losses = []
        train_acc = []
        val_acc = []

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            running_acc = 0.0

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
                    print(f"Epoch {epoch + 1} Batch {i} Loss {loss.item():.4f} Acc {acc:.4f}")

            self.scheduler.step()
            train_losses.append(running_loss / len(train_loader))
            train_acc.append(running_acc / len(train_loader))

            val_accuracy = self.validate(val_loader)
            val_acc.append(val_accuracy)
            print(f"Validation Accuracy: {val_accuracy:.4f}")

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

    def export_onnx(self, path="imagenet_model.onnx"):
        dummy = torch.randn(1, 3, 224, 224).to(self.device)

        self.model.eval()
        torch.onnx.export(
            self.model,
            dummy,
            path,
            opset_version=17,
            input_names=["images"],
            output_names=["logits"],
            dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
        )

        print("ONNX model saved to", path)
