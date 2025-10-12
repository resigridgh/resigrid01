import unittest
import torch
import numpy as np
import sys
import os

# Add package to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from mypythonpackage.model.regression import LinearRegression, run_hydropower


class TestLinearRegression(unittest.TestCase):
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Create synthetic data: y = 2 + 3x + noise
        n_samples = 100
        self.X_train = np.random.randn(n_samples).astype(np.float32)
        noise = np.random.randn(n_samples) * 0.1
        self.y_train = 2.0 + 3.0 * self.X_train + noise
        
        self.model = LinearRegression(learning_rate=0.01, epochs=100)
    
    def test_initialization(self):
        """Test model initialization"""
        self.assertEqual(self.model.learning_rate, 0.01)
        self.assertEqual(self.model.epochs, 100)
        self.assertTrue(self.model.w0.requires_grad)
        self.assertTrue(self.model.w1.requires_grad)
    
    def test_forward(self):
        """Test forward pass"""
        x_test = torch.tensor([1.0, 2.0], dtype=torch.float32)
        output = self.model.forward(x_test)
        self.assertEqual(output.shape, (2,))
    
    def test_fit_predict(self):
        """Test training and prediction"""
        self.model.fit(self.X_train, self.y_train)
        predictions = self.model.predict(np.array([1.0, 2.0]))
        self.assertEqual(predictions.shape, (2,))
    
    def test_hydropower(self):
    
        try:
            model, scaler, df = run_hydropower()
            success = True
        except Exception as e:
            success = False
            print(f"Hydropower example failed: {e}")
        
        self.assertTrue(success)


if __name__ == '__main__':
    unittest.main()
