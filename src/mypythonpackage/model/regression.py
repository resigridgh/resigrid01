"""
regression.py
=============

Implements Linear Regression in one variable using PyTorch.
Provides functions to train the model using Stochastic Gradient Descent (SGD),
make predictions, evaluate R², and visualize results.

Author : Md Saifu Islam
Email  : mi1499@uah.edu
"""

import torch
from torch import nn
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


class LinearRegression:
    """
    Linear Regression model using PyTorch (1 variable).

    Model equation:
        y = w0 + w1 * x

    Parameters
    ----------
    lr : float
        Learning rate (default = 0.001)
    epochs : int
        Number of training iterations (default = 1000)
    device : str, optional
        'cuda' or 'cpu' (automatically chosen if not provided)
    """

    def __init__(self, lr=1e-3, epochs=1000, device=None):
        self.lr = lr
        self.epochs = epochs
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model parameters (weights)
        self.w0 = torch.randn(1, requires_grad=True, dtype=torch.float32, device=self.device)
        self.w1 = torch.randn(1, requires_grad=True, dtype=torch.float32, device=self.device)

        # Define optimizer and loss function
        self.optimizer = torch.optim.SGD([self.w0, self.w1], lr=self.lr)
        self.loss_fn = nn.MSELoss()

        # Store training history for analysis
        self.history = {"loss": [], "w0": [], "w1": []}

    # --------------------------------------------------------
    def forward(self, x):
        """Compute model output y_pred = w0 + w1 * x"""
        x = x.view(-1).to(self.device).float()
        return self.w0 + self.w1 * x

    # --------------------------------------------------------
    def fit(self, X_train, y_train, X_test=None, y_test=None, verbose=False):
        """
        Train the model using Stochastic Gradient Descent (SGD).

        Parameters
        ----------
        X_train, y_train : array-like
            Training data
        X_test, y_test : array-like, optional
            Test data (for computing R² after training)
        verbose : bool
            If True, print progress every 10% of training
        """

        # Convert to PyTorch tensors
        X_train_t = torch.tensor(np.array(X_train), dtype=torch.float32).to(self.device)
        y_train_t = torch.tensor(np.array(y_train), dtype=torch.float32).to(self.device)

        for epoch in range(self.epochs):
            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            y_pred = self.forward(X_train_t)

            # Compute loss
            loss = self.loss_fn(y_pred, y_train_t)

            # Backward pass
            loss.backward()

            # Update weights
            self.optimizer.step()

            # Save history
            self.history["loss"].append(loss.item())
            self.history["w0"].append(self.w0.detach().cpu().item())
            self.history["w1"].append(self.w1.detach().cpu().item())

            # Print progress
            if verbose and ((epoch + 1) % max(1, self.epochs // 10) == 0):
                print(f"Epoch {epoch+1}/{self.epochs} - Loss: {loss.item():.6f}")

        # Evaluate R² on test data if provided
        if X_test is not None and y_test is not None:
            y_pred_test = self.predict(X_test)
            r2 = r2_score(y_test, y_pred_test)
            print(f"\nR² score on test data: {r2:.6f}")
            return r2

    # --------------------------------------------------------
    def predict(self, X_new):
        """Predict outputs for new input data."""
        X_t = torch.tensor(np.array(X_new), dtype=torch.float32).to(self.device)
        y_t = self.forward(X_t)
        return y_t.detach().cpu().numpy()

    # --------------------------------------------------------
    def summary(self):
        """Print model parameters."""
        print(f"w0 (intercept): {self.w0.detach().cpu().item():.6f}")
        print(f"w1 (slope)    : {self.w1.detach().cpu().item():.6f}")

    # --------------------------------------------------------
    def plot_results(self, X, y, save_path=None, dpi=300):
        """
        Plot data points, fitted line, and training loss curve.

        Parameters
        ----------
        X, y : array-like
            Input and output data
        save_path : str
            Optional path to save figure (PDF or PNG)
        dpi : int
            Resolution of the saved figure (default 300)
        """
        X = np.array(X).flatten()
        y = np.array(y).flatten()

        # Prediction line (sorted X for smoothness)
        sort_idx = np.argsort(X)
        X_sorted = X[sort_idx]
        y_pred_sorted = self.predict(X_sorted)

        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        # Plot data and fitted line
        axs[0].scatter(X, y, label="Data", color="blue", alpha=0.7)
        axs[0].plot(X_sorted, y_pred_sorted, color="red", label="Fitted Line", linewidth=2)
        axs[0].set_xlabel("Independent Variable (X)")
        axs[0].set_ylabel("Dependent Variable (Y)")
        axs[0].set_title("Linear Regression Fit")
        axs[0].legend()
        axs[0].grid(True, linestyle="--", alpha=0.6)

        # Plot loss curve
        axs[1].plot(self.history["loss"], color="green", linewidth=1.5)
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("MSE Loss")
        axs[1].set_title("Training Loss Curve")
        axs[1].grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=dpi, format="pdf")
        plt.show()

