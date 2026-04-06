import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from mypythonpackage.deepl import ACCDataset, ACCNet


DATA_DIR = "/data/CPE_487-587/ACCDataset"

BATCH_SIZE = 64
EPOCHS = 100
LR = 0.005

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


dataset = ACCDataset(DATA_DIR, k=10)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)


model = ACCNet().to(device)

criterion = torch.nn.BCEWithLogitsLoss()

optimizer = optim.Adam(model.parameters(), lr=LR)


train_losses = []
val_accuracies = []


for epoch in range(EPOCHS):

    model.train()

    total_loss = 0
    batch_count = 0

    for X, y in train_loader:

        X = X.to(device)
        y = y.to(device)

        preds = model(X)

        loss = criterion(preds, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batch_count += 1

    avg_loss = total_loss / batch_count
    train_losses.append(avg_loss)


    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for X, y in val_loader:

            X = X.to(device)
            y = y.to(device)

            preds = model(X)

            probs = torch.sigmoid(preds)

            predicted = (probs > 0.5).float()

            correct += (predicted == y).sum().item()
            total += y.size(0)

    acc = correct / total
    val_accuracies.append(acc)

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train Loss: {avg_loss:.4f} | "
        f"Val Acc: {acc:.4f}"
    )


plt.figure()
plt.plot(train_losses)
plt.title("Training Loss vs Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig("loss_vs_epochs.png")


plt.figure()
plt.plot(val_accuracies)
plt.title("Validation Accuracy vs Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)
plt.savefig("accuracy_vs_epochs.png")

print("Plots saved")


dummy_input = torch.randn(1, 11).to(device)

torch.onnx.export(
    model,
    dummy_input,
    "acc_model.onnx",
    input_names=["speed_history"],
    output_names=["acc_state"],
    opset_version=18
)

print("ONNX model saved")
