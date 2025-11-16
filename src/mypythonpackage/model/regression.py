import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')


class LinearRegression:
    
    def __init__(self, learning_rate=0.01, epochs=1000):
    
        self.learning_rate = learning_rate
        self.epochs = epochs
        
        # Initialize parameters w0 (bias) and w1 (weight)
        self.w0 = torch.tensor(0.0, requires_grad=True, dtype=torch.float32)
        self.w1 = torch.tensor(0.0, requires_grad=True, dtype=torch.float32)
        
        # SGD optimizer
        self.optimizer = torch.optim.SGD([self.w0, self.w1], lr=learning_rate)
        
        # Mean-squared error loss function
        self.criterion = nn.MSELoss()
        
        # Lists to store intermediate values
        self.w0_history = []
        self.w1_history = []
        self.loss_history = []
        
        print(f"LinearRegression initialized with learning_r

        print(f"Training started with {len(X_train)} samples")
        print(f"Initial parameters: w0 = {self.w0.item():.4f}, w1 = {self.w1.item():.4f}")
        
        # Training loop
        for epoch in range(self.epochs):
            # Zero gradients
            self.optimi

            with torch.no_grad():
                y_pred_test = self.forward(X_test)
                r2 = r2_score(y_test.numpy(), y_pred_test.numpy())
                print(f'R² Score on Test Data: {r2:.6f}')
    
    def predict(self, x):

        # Convert to tensor if needed
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        
        # Ensure proper dimensions
        if x.dim() > 1 and x.shape[1] == 1:
            x = x.squeeze()
        
        # Predict without tracking gradients
        with torch.no_grad():
            predictions = self.forward(x)
        
        return predictions.numpy()
    
    def plot_analysis(self, X_train=None, y_train=None):
    
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Original data and fitted regression line
        if X_train is not None and y_train is not None:
            # Convert to numpy for plotting
            if isinstance(X_train, torch.Tensor):
                X_plot = X_train.numpy()
            else:
                X_plot = X_train
                
            if isinstance(y_train, torch.Tensor):
                y_plot = y_train.numpy()
            else:
                y_plot = y_train
            
            axes[0, 0].scatter(X_plot, y_plot, alpha=0.6, color='blue', label='Training Data')
            
            # Create regression line
            x_min, x_max = np.min(X_plot), np.max(X_plot)
            x_line = np.linspace(x_min, x_max, 100)
            y_line = self.predict(x_line)
            
            axes[0, 0].plot(x_line, y_line, 'r-', linewidth=2, label=f'Regression Line: y = {self.w0.item():.4f} + {self.w1.item():.4f}x')
            axes[0, 0].set_xlabel('Benefit-Cost-Ratio (BCR)')
            axes[0, 0].set_ylabel('Annual Production')
            axes[0, 0].set_title('Linear Regression Fit')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: w0 (bias) history during training
        axes[0, 1].plot(self.w0_history, 'g-', linewidth=1)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('w0 (Bias)')
        axes[0, 1].set_title('Bias (w0) during Training')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: w1 (weight) history during training
        axes[1, 0].plot(self.w1_history, 'orange', linewidth=1)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('w1 (Weight)')
        axes[1, 0].set_title('Weight (w1) during Training')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Loss history during training
        axes[1, 1].plot(self.loss_history, 'r-', linewidth=1)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss (MSE)')
        axes[1, 1].set_title('Loss during Training')
        axes[1, 1].set_yscale('log')  # Log scale for better visualization
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print final model equation
        print(f"\nFinal Linear Model: AnnualProduction = {self.w0.item():.6f} + {self.w1.item():.6f} * BCR")


def run_hydropower():

    print("=" * 70)
    print("HYDROPOWER DATASET - LinearRegression Class")
    print("=" * 70)
    
    # Load the dataset
    url = "https://raw.githubusercontent.com/rahulbhadani/CPE486586_FA25/main/Data/Hydropower.csv"
    
    try:
        df = pd.read_csv(url)
        print(" Dataset loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print(" Creating synthetic data for demonstration...")
        np.random.seed(42)
        n_samples = 200
        bcr = np.random.uniform(0.5, 3.0, n_samples)
        annual_production = 1000 + 500 * bcr + np.random.normal(0, 100, n_samples)
        df = pd.DataFrame({
            'Benefit-Cost-Ratio (BCR)': bcr,
            'AnnualProduction': annual_production
        })
    
    print(f"\n📊 Dataset shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Prepare data
    X = df['Benefit-Cost-Ratio (BCR)'].values.reshape(-1, 1)
    y = df['AnnualProduction'].values
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\n Data split:")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).flatten()
    X_test_scaled = scaler.transform(X_test).flatten()
    
    print(f"\n Feature scaling applied")
    
    # Create and train the model
    print("\n" + "=" * 50)
    print(" MODEL TRAINING")
    print("=" * 50)
    
    model = LinearRegression(learning_rate=0.01, epochs=1000)
    model.fit(X_train_scaled, y_train, X_test_scaled, y_test)
    
    # Make predictions
    print("\n" + "=" * 50)
    print("PREDICTIONS")
    print("=" * 50)
    
    new_bcr_values = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
    new_bcr_scaled = scaler.transform(new_bcr_values.reshape(-1, 1)).flatten()
    predictions = model.predict(new_bcr_scaled)
    
    print("\nPredictions for new BCR values:")
    for bcr, pred in zip(new_bcr_values, predictions):
        print(f"BCR: {bcr:.1f} -> Predicted Annual Production: {pred:.2f}")
    
    # Test set performance
    print("\n" + "=" * 50)
    print("FINAL PERFORMANCE")
    print("=" * 50)
    
    y_pred_test = model.predict(X_test_scaled)
    test_r2 = r2_score(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    print(f"Test R²: {test_r2:.6f}")
    print(f"Test RMSE: {test_rmse:.6f}")
    
    # Convert to original scale
    w1_original = model.w1.item() / scaler.scale_[0]
    w0_original = model.w0.item() - w1_original * scaler.mean_[0]
    
    print(f"\n📝 FINAL MODEL (Original Scale):")
    print(f"AnnualProduction = {w0_original:.6f} + {w1_original:.6f} * BCR")
    
    # Generate analysis plots
    print("\n" + "=" * 50)

run_hydropower()


class CauchyRegression:

    def __init__(self, input_dim=4, c=1.0, lr=1e-3, epochs=1000,
                 weight_decay=0.0, device='cpu', verbose=True):
        self.input_dim = input_dim
        self.c = float(c)
        self.lr = lr
        self.epochs = int(epochs)
        self.device = torch.device(device)
        self.verbose = verbose

        # Model parameters
        self.w = nn.Parameter(
            torch.randn((input_dim, 1), dtype=torch.float32, device=self.device) * 0.1
        )
        self.b = nn.Parameter(
            torch.zeros(1, dtype=torch.float32, device=self.device)
        )

        # Optimizer
        self.optimizer = torch.optim.Adam(
            [self.w, self.b], lr=self.lr, weight_decay=weight_decay
        )

        # Training history
        self.history = {
            'train_loss': [],
            'test_loss': []
        }

    def forward(self, X):
    
        return X @ self.w + self.b

    def cauchy_loss(self, y_pred, y_true):

        residuals = y_true - y_pred
        scaled_residuals = residuals / self.c
        loss = 0.5 * (self.c ** 2) * torch.log(1 + scaled_residuals ** 2)
        return loss.mean()

    def fit(self, X_train, y_train, X_test=None, y_test=None):
        
        print("Training Cauchy Regression Model...")

        for epoch in range(1, self.epochs + 1):
            # Training step
            self.optimizer.zero_grad()
            y_pred_train = self.forward(X_train)
            train_loss = self.cauchy_loss(y_pred_train, y_train)
            train_loss.backward()
            self.optimizer.step()

            # Store training loss
            self.history['train_loss'].append(train_loss.item())

            # Calculate test loss if test data provided
            if X_test is not None and y_test is not None:
                with torch.no_grad():
                    y_pred_test = self.forward(X_test)
                    test_loss = self.cauchy_loss(y_pred_test, y_test)
                self.history['test_loss'].append(test_loss.item())

            # Progress reporting
            if self.verbose and (epoch <= 5 or epoch % max(1, self.epochs//10) == 0):
                if X_test is not None and y_test is not None:
                    print(f"Epoch {epoch:4d}/{self.epochs} | "
                          f"Train Loss: {train_loss.item():.4f} | "
                          f"Test Loss: {test_loss.item():.4f}")
                else:
                    print(f"Epoch {epoch:4d}/{self.epochs} | "
                          f"Train Loss: {train_loss.item():.4f}")

        return self.history

    def predict(self, X):
        """Make predictions on tensor data."""
        with torch.no_grad():
            return self.forward(X)

    def predict_numpy(self, X_np):
        """Make predictions on numpy array."""
        X_tensor = torch.tensor(X_np, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            predictions = self.forward(X_tensor).cpu().numpy()
        return predictions

    def get_parameters(self):
        """Get model parameters as numpy arrays."""
        w_np = self.w.detach().cpu().numpy().flatten()
        b_np = float(self.b.detach().cpu().numpy())
        return b_np, w_np

    def transform_to_original_scale(self, scaler):
        """Transform coefficients back to original feature scale."""
        b_scaled, w_scaled = self.get_parameters()
        w_original = w_scaled / scaler.scale_
        b_original = b_scaled - np.dot(scaler.mean_, w_scaled / scaler.scale_)
        return b_original, w_original

    def score(self, X, y, metric='cauchy'):
        """Calculate model score on given data."""
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            y_pred = self.forward(X)
            if metric == 'cauchy':
                return self.cauchy_loss(y_pred, y).item()
            elif metric == 'mse':
                return torch.mean((y_pred - y) ** 2).item()
            else:
                raise ValueError("metric must be 'cauchy' or 'mse'")
