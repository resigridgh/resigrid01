import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import requests
from io import StringIO
import onnxruntime as ort

plt.rcParams['font.family'] = 'Serif'
plt.rcParams['font.size'] = 15

class TorchNet:
    """
    A neural network class for binary classification tasks.
    
    This class provides a complete pipeline for loading data, training a neural network,
    evaluating performance, and exporting the model to ONNX format.
    
    Attributes:
        input_dim (int): Number of input features
        output_dim (int): Number of output classes (1 for binary classification)
        model (nn.Sequential): PyTorch neural network model
        scaler (StandardScaler): Scaler for data standardization
        losses (list): Training loss history
    """
    
    def __init__(self, input_dim, output_dim):
        """
        Initialize the TorchNet model.
        
        Args:
            input_dim (int): Number of input features
            output_dim (int): Number of output classes
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, 4),
            nn.ReLU(),
            nn.Linear(4, 3),
            nn.ReLU(),
            nn.Linear(3, output_dim),
            nn.Sigmoid()
        )
        self.scaler = StandardScaler()
        self.losses = []

    def load_data(self, data):
        """
        Load and preprocess the dataset.
        
        Args:
            data (pd.DataFrame): Input dataset
            
        Returns:
            tuple: Contains training and testing data as PyTorch tensors
                   (X_train, X_test, y_train, y_test)
        """
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        return (
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32).view(-1, 1),
            torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
        )

    def train_model(self, X_train, y_train, epochs=1000, lr=0.01):
        """
        Train the neural network model.
        
        Args:
            X_train (torch.Tensor): Training features
            y_train (torch.Tensor): Training labels
            epochs (int): Number of training epochs
            lr (float): Learning rate
        """
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        print("Training started")
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            self.losses.append(loss.item())

            if (epoch+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    def plot_loss(self):
        """Plot the training loss curve."""
        plt.figure(figsize=(10, 5))
        plt.plot(self.losses, color='red', linewidth=2)
        plt.title('Training Loss vs Epochs', pad=20)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')

        ax = plt.gca()
        ax.spines['bottom'].set_color('black')
        ax.spines['top'].set_color('black')
        ax.spines['left'].set_color('black')
        ax.spines['right'].set_color('black')
        ax.tick_params(direction='out', color='black')

        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the model on test data.
        
        Args:
            X_test (torch.Tensor): Test features
            y_test (torch.Tensor): Test labels
            
        Returns:
            float: Test accuracy
        """
        with torch.no_grad():
            outputs = self.model(X_test)
            predicted = (outputs >= 0.5).float()
            accuracy = (predicted == y_test).float().mean()
            return accuracy.item()

    def save_model(self, file_path):
        """
        Save the model in ONNX format and scaler parameters.
        
        Args:
            file_path (str): Path to save the ONNX model
        """
        dummy_input = torch.randn(1, self.input_dim)
        input_names = ["input"]
        output_names = ["output"]

        torch.onnx.export(
            self.model, dummy_input, file_path,
            input_names=input_names, output_names=output_names,
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            opset_version=11
        )
        print(f"Model saved as {file_path}")

        # Save scaler parameters
        scaler_params = {
            "mean": self.scaler.mean_.tolist(),
            "scale": self.scaler.scale_.tolist()
        }
        with open("scaler_params.json", "w") as f:
            json.dump(scaler_params, f)
        print("Scaler parameters saved as scaler_params.json")

    def predict_onnx(self, new_data, model_path='diabetes_model.onnx'):
        """
        Make predictions using the saved ONNX model.
        
        Args:
            new_data (list or np.array): New data for prediction
            model_path (str): Path to the ONNX model file
            
        Returns:
            tuple: (prediction, prediction_label)
        """
        # Load scaler parameters
        with open("scaler_params.json", "r") as f:
            scaler_params = json.load(f)

        new_data_array = np.array(new_data, dtype=np.float32)
        new_data_scaled = (new_data_array - scaler_params["mean"]) / scaler_params["scale"]

        # Load ONNX model and make prediction
        ort_session = ort.InferenceSession(model_path)
        input_name = ort_session.get_inputs()[0].name
        onnx_input = new_data_scaled.astype(np.float32)

        # Run prediction using loaded ONNX model
        onnx_output = ort_session.run(None, {input_name: onnx_input})
        onnx_prediction = (onnx_output[0] >= 0.5).astype(int)
        
        prediction_label = "Diabetes" if onnx_prediction[0][0] == 1 else "No Diabetes"
        
        return onnx_prediction[0][0], prediction_label
