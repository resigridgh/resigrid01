import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn as nn


# =========================================================
# Dataset
# =========================================================

class ACCDataset(Dataset):
    def __init__(self, data_dir, k=10, transform=None):
        self.k = k
        self.transform = transform

        speed_files = sorted(
            glob.glob(os.path.join(data_dir, "*decoded_wheel_speed_fl.csv"))
        )

        status_file = os.path.join(data_dir, "acc_status.csv")

        self.X = []
        self.y = []

        status_df = pd.read_csv(status_file)

        for file in speed_files:

            speed_df = pd.read_csv(file)

            if "Message" not in speed_df.columns:
                continue

            speeds = speed_df["Message"].values

            labels = status_df["acc_status"].values

            n = min(len(speeds), len(labels))

            speeds = speeds[:n]
            labels = labels[:n]

            for t in range(k, n):

                feature = speeds[t-k:t+1]

                label = 1 if labels[t] == 6 else 0

                self.X.append(feature)
                self.y.append(label)

        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        x = self.X[idx]
        y = self.y[idx]

        if self.transform:
            x = self.transform(x)

        return x, y


# =========================================================
# Neural Network
# =========================================================

class ACCNet(nn.Module):

    def __init__(self, input_dim=11):

        super().__init__()

        self.model = nn.Sequential(

            nn.Linear(input_dim, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


# =========================================================
# Dice Loss
# =========================================================

class DiceLoss(nn.Module):

    def __init__(self, smooth=1):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):

        preds = preds.view(-1)
        targets = targets.view(-1)

        intersection = (preds * targets).sum()

        dice = (2. * intersection + self.smooth) / (
            preds.sum() + targets.sum() + self.smooth
        )

        return 1 - dice
