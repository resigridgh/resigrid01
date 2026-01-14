import torch
import torch.nn as nn

def binary_classification(d, n, epochs=10000, lr=0.001):
    
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Generate random feature matrix X
    X = torch.randn(n, d, dtype=torch.float32, device=device)
    
    # 2. Generate labels Y: if sum of features > 2 -> 1, else 0
    Y = (X.sum(dim=1, keepdim=True) > 2).float().to(device)
    
    # 3. Initialize weights with He initialization
    def init_weights(fan_in):
        std = torch.sqrt(torch.tensor(2.0 / fan_in))
        return torch.randn(fan_in, dtype=torch.float32, device=device) * std
    
    # Dimensions
    W1 = torch.randn(d, 48, dtype=torch.float32, device=device) * torch.sqrt(torch.tensor(2.0 / d))
    W2 = torch.randn(48, 16, dtype=torch.float32, device=device) * torch.sqrt(torch.tensor(2.0 / 48))
    W3 = torch.randn(16, 32, dtype=torch.float32, device=device) * torch.sqrt(torch.tensor(2.0 / 16))
    W4 = torch.randn(32, 1, dtype=torch.float32, device=device) * torch.sqrt(torch.tensor(2.0 / 32))
    
    # Enable gradient tracking
    W1.requires_grad_(True)
    W2.requires_grad_(True)
    W3.requires_grad_(True)
    W4.requires_grad_(True)
    
    # Sigmoid activation
    sigmoid = nn.Sigmoid()
    
    # Loss history
    loss_history = []
    
    # Training loop
    for epoch in range(epochs):
        # Forward pass
        Z1 = torch.matmul(X, W1)
        A1 = sigmoid(Z1)
        
        Z2 = torch.matmul(A1, W2)
        A2 = sigmoid(Z2)
        
        Z3 = torch.matmul(A2, W3)
        A3 = sigmoid(Z3)
        
        Z4 = torch.matmul(A3, W4)
        Y_pred = sigmoid(Z4)
        
        # Binary Cross Entropy Loss
        loss = -torch.mean(Y * torch.log(Y_pred + 1e-8) + (1 - Y) * torch.log(1 - Y_pred + 1e-8))
        
        # Backward pass
        loss.backward()
        
        # Gradient descent update (no optimizer, manual update)
        with torch.no_grad():
            W1 -= lr * W1.grad
            W2 -= lr * W2.grad
            W3 -= lr * W3.grad
            W4 -= lr * W4.grad
            
            # Zero gradients
            W1.grad.zero_()
            W2.grad.zero_()
            W3.grad.zero_()
            W4.grad.zero_()
        
        # Record loss
        loss_history.append(loss.item())
        
        if (epoch + 1) % 1000 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
    
    print("Training completed.")
    
    # Detach and return weights (no gradient needed for return)
    return W1.detach(), W2.detach(), W3.detach(), W4.detach(), loss_history
