import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from deepl.acc_classifier import ACCDataset, ACCNet, DiceLoss


# =====================================================
# Config
# =====================================================

DATA_DIR = "/data/CPE_487-587/ACCDataset"

BATCH_SIZE = 64
EPOCHS = 20
LR = 0.001

MODEL_OUT = "acc_model.onnx"


# =====================================================
# Device
# =====================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


# =====================================================
# Dataset
# =====================================================

dataset = ACCDataset(DATA_DIR, k=10)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)


# =====================================================
# Model
# =====================================================

model = ACCNet(input_dim=11).to(device)

criterion = DiceLoss()

optimizer = optim.Adam(model.parameters(), lr=LR)


# =====================================================
# Training Loop
# =====================================================

for epoch in range(EPOCHS):

    model.train()

    train_loss = 0

    for X, y in train_loader:

        X = X.to(device)
        y = y.to(device)

        preds = model(X)

        loss = criterion(preds, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for X, y in val_loader:

            X = X.to(device)
            y = y.to(device)

            preds = model(X)

            predicted = (preds > 0.5).float()

            correct += (predicted == y).sum().item()
            total += y.size(0)

    acc = correct / total

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Acc: {acc:.4f}"
    )


# =====================================================
# Export ONNX
# =====================================================

model.eval()

dummy_input = torch.randn(1, 11).to(device)

torch.onnx.export(
    model,
    dummy_input,
    MODEL_OUT,
    input_names=["speed_history"],
    output_names=["acc_state"],
    opset_version=11
)

print("ONNX model saved:", MODEL_OUT)
