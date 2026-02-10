from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

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
        super(SimpleNN, self).__init__()
        if in_features <= 0:
            raise ValueError("in_features must be > 0")
        if num_classes <= 1:
            raise ValueError("num_classes must be >= 2 for multi-class classification")

        self.in_features = in_features
        self.num_classes = num_classes

        self.fc1 = nn.Linear(self.in_features, 3)
        self.fc2 = nn.Linear(3, 4)
        self.fc3 = nn.Linear(4, 5)
        self.fc4 = nn.Linear(5, self.num_classes)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)  # logits (N, num_classes)
        return x


@dataclass
class EvalMetrics:
    accuracy: float
    f1: float
    precision: float
    recall: float


class ClassTrainer:
    """
    Trainer for multi-class classification.

    Required class variables from HW02Q8:
      X_train, y_train, eta, epoch, loss, optimizer, loss_vector, accuracy_vector,
      model, device
    """

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
        # Reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.X_train = X_train.to(self.device)
        self.y_train = y_train.to(self.device)

        self.eta = float(eta)
        self.epoch = int(epoch)
        self.batch_size = int(batch_size)

        if model is None:
            raise ValueError("model must be provided (e.g., SimpleNN(in_features, num_classes))")
        self.model = model.to(self.device)

        # Default loss for multi-class logits
        self.loss = loss if loss is not None else nn.CrossEntropyLoss()

        opt_name = optimizer_name.strip().lower()
        if opt_name == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.eta)
        elif opt_name == "sgd":
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.eta, momentum=0.9)
        else:
            raise ValueError("optimizer_name must be 'adam' or 'sgd'")

        # loss_vector and accuracy_vector: torch tensors length = epochs
        self.loss_vector = torch.zeros(self.epoch, device=self.device)
        self.accuracy_vector = torch.zeros(self.epoch, device=self.device)

        # For evaluation/plots
        self._train_true: Optional[np.ndarray] = None
        self._train_pred: Optional[np.ndarray] = None
        self._test_true: Optional[np.ndarray] = None
        self._test_pred: Optional[np.ndarray] = None

    def _make_batches(self, X: torch.Tensor, y: torch.Tensor):
        n = X.shape[0]
        idx = torch.randperm(n, device=X.device)
        for start in range(0, n, self.batch_size):
            batch_idx = idx[start:start + self.batch_size]
            yield X[batch_idx], y[batch_idx]

    def train(self) -> None:
        self.model.train()

        for ep in range(self.epoch):
            epoch_losses = []
            all_preds = []
            all_true = []

            for xb, yb in self._make_batches(self.X_train, self.y_train):
                self.optimizer.zero_grad(set_to_none=True)

                logits = self.model(xb)  # (B, C)
                loss_val = self.loss(logits, yb.long())
                loss_val.backward()
                self.optimizer.step()

                epoch_losses.append(loss_val.detach())

                preds = torch.argmax(logits.detach(), dim=1)
                all_preds.append(preds)
                all_true.append(yb.detach())

            # record epoch loss + accuracy
            mean_loss = torch.stack(epoch_losses).mean()
            y_true = torch.cat(all_true).cpu().numpy()
            y_pred = torch.cat(all_preds).cpu().numpy()

            acc = accuracy_score(y_true, y_pred)

            self.loss_vector[ep] = mean_loss
            self.accuracy_vector[ep] = torch.tensor(acc, device=self.device)

        # Store final train preds for evaluation()
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.X_train)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            true = self.y_train.cpu().numpy()
        self._train_true, self._train_pred = true, preds

    def test(self, X_test: torch.Tensor, y_test: torch.Tensor) -> EvalMetrics:
        self.model.eval()
        X_test = X_test.to(self.device)
        y_test = y_test.to(self.device)

        with torch.no_grad():
            logits = self.model(X_test)
            preds = torch.argmax(logits, dim=1)

        y_true = y_test.cpu().numpy()
        y_pred = preds.cpu().numpy()

        self._test_true, self._test_pred = y_true, y_pred

        return EvalMetrics(
            accuracy=float(accuracy_score(y_true, y_pred)),
            f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            precision=float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            recall=float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        )

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        X = X.to(self.device)
        with torch.no_grad():
            logits = self.model(X)
            preds = torch.argmax(logits, dim=1)
        return preds

    def save(self, file_name: Optional[str] = None) -> str:
        """
        Save model to ONNX.
        """
        import os
        from datetime import datetime

        if file_name is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"multiclass_model_{ts}.onnx"

        os.makedirs(os.path.dirname(file_name) or ".", exist_ok=True)

        self.model.eval()
        dummy = torch.randn(1, getattr(self.model, "in_features", self.X_train.shape[1]), device=self.device)

        torch.onnx.export(
            self.model,
            dummy,
            file_name,
            input_names=["X"],
            output_names=["logits"],
            dynamic_axes={"X": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=17,
        )
        return file_name

    def evaluation(
        self,
        loss_vector: Optional[torch.Tensor] = None,
        accuracy_vector: Optional[torch.Tensor] = None,
        class_names: Optional[list[str]] = None,
        out_dir: str = "outputs/plots",
        tag: str = "run",
    ) -> Dict[str, Any]:
        """
        Plots:
          1) Training loss curve
          2) Training accuracy curve
          3) Confusion matrix (train)
          4) Confusion matrix (test, if test() was called)

        Also prints + returns final metrics for train and test (if available).
        """
        import os
        from datetime import datetime

        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        lv = self.loss_vector if loss_vector is None else loss_vector
        av = self.accuracy_vector if accuracy_vector is None else accuracy_vector

        # Loss plot
        plt.figure()
        plt.plot(lv.detach().cpu().numpy())
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        loss_path = os.path.join(out_dir, f"loss_{tag}_{ts}.png")
        plt.savefig(loss_path, bbox_inches="tight")
        plt.close()

        # Accuracy plot
        plt.figure()
        plt.plot(av.detach().cpu().numpy())
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Training Accuracy")
        acc_path = os.path.join(out_dir, f"accuracy_{tag}_{ts}.png")
        plt.savefig(acc_path, bbox_inches="tight")
        plt.close()

        results: Dict[str, Any] = {
            "loss_plot": loss_path,
            "accuracy_plot": acc_path,
        }

        # Train metrics + CM
        if self._train_true is not None and self._train_pred is not None:
            tr_true = self._train_true
            tr_pred = self._train_pred

            train_metrics = EvalMetrics(
                accuracy=float(accuracy_score(tr_true, tr_pred)),
                f1=float(f1_score(tr_true, tr_pred, average="macro", zero_division=0)),
                precision=float(precision_score(tr_true, tr_pred, average="macro", zero_division=0)),
                recall=float(recall_score(tr_true, tr_pred, average="macro", zero_division=0)),
            )

            cm = confusion_matrix(tr_true, tr_pred)
            disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
            disp.plot()
            plt.title("Confusion Matrix (Train)")
            cm_train_path = os.path.join(out_dir, f"cm_train_{tag}_{ts}.png")
            plt.savefig(cm_train_path, bbox_inches="tight")
            plt.close()

            results["train_metrics"] = train_metrics
            results["cm_train_plot"] = cm_train_path

        # Test metrics + CM (only if test() called)
        if self._test_true is not None and self._test_pred is not None:
            te_true = self._test_true
            te_pred = self._test_pred

            test_metrics = EvalMetrics(
                accuracy=float(accuracy_score(te_true, te_pred)),
                f1=float(f1_score(te_true, te_pred, average="macro", zero_division=0)),
                precision=float(precision_score(te_true, te_pred, average="macro", zero_division=0)),
                recall=float(recall_score(te_true, te_pred, average="macro", zero_division=0)),
            )

            cm = confusion_matrix(te_true, te_pred)
            disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
            disp.plot()
            plt.title("Confusion Matrix (Test)")
            cm_test_path = os.path.join(out_dir, f"cm_test_{tag}_{ts}.png")
            plt.savefig(cm_test_path, bbox_inches="tight")
            plt.close()

            results["test_metrics"] = test_metrics
            results["cm_test_plot"] = cm_test_path

        return results
