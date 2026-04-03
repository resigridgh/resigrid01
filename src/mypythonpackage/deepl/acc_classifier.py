import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class ACCDataset(Dataset):
    """
    Dataset for Adaptive Cruise Control classification
    """

    def __init__(self, data_dir, k=10):

        self.X = []
        self.y = []

        speed_files = glob.glob(
            os.path.join(data_dir, "*decoded_wheel_speed_fl.csv")
        )

        print("Found wheel speed files:", len(speed_files))

        for speed_file in speed_files:

            status_file = speed_file.replace(
                "decoded_wheel_speed_fl",
                "decoded_acc_status"
            )

            if not os.path.exists(status_file):
                continue

            speed_df = pd.read_csv(speed_file)
            status_df = pd.read_csv(status_file)

            speeds = speed_df["Message"].values
            status = status_df["Message"].values

            n = min(len(speeds), len(status))

            for t in range(k, n):

                # vt, vt-1, ..., vt-10
                feature = speeds[t-k:t+1][::-1]

                label = 1 if status[t] == 6 else 0

                self.X.append(feature)
                self.y.append(label)

        self.X = np.array(self.X)

        # normalization
        self.mean = self.X.mean()
        self.std = self.X.std()

        self.X = (self.X - self.mean) / self.std

        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32).unsqueeze(1)

        print("Total dataset samples:", len(self.X))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        return self.X[idx], self.y[idx]


class ACCNet(nn.Module):
    """
    1D CNN for time-series ACC prediction
    """

    def __init__(self):

        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(

            nn.Linear(64 * 11, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, x):

        x = x.unsqueeze(1)   # shape: (batch, 1, 11)

        x = self.conv(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x
