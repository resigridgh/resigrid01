import torch
import torch.nn as nn

def binary_classification(d, n, epochs=10000, lr=0.001, store_on_cpu=True):
    """
    HW02 version:
    - Trains same network as HW01
    - Stores W1..W4 snapshots at every epoch as 3D tensors:
        W1_hist: [epochs, d, 48]
        W2_hist: [epochs, 48, 16]
        W3_hist: [epochs, 16, 32]
        W4_hist: [epochs, 32, 1]
    - Returns: W1_hist, W2_hist, W3_hist, W4_hist, loss_history
    """

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    X = torch.randn(n, d, dtype=torch.float32, device=device)
    Y = (X.sum(dim=1, keepdim=True) > 2).float().to(device)

    # He init
    W1 = torch.randn(d, 48, dtype=torch.float32, device=device) * torch.sqrt(torch.tensor(2.0 / d, device=device))
    W2 = torch.randn(48, 16, dtype=torch.float32, device=device) * torch.sqrt(torch.tensor(2.0 / 48, device=device))
    W3 = torch.randn(16, 32, dtype=torch.float32, device=device) * torch.sqrt(torch.tensor(2.0 / 16, device=device))
    W4 = torch.randn(32, 1, dtype=torch.float32, device=device) * torch.sqrt(torch.tensor(2.0 / 32, device=device))

    # Grad tracking
    W1.requires_grad_(True)
    W2.requires_grad_(True)
    W3.requires_grad_(True)
    W4.requires_grad_(True)

    sigmoid = nn.Sigmoid()
    loss_history = []

    # HW02: preallocate weight histories (3D stacks)
    hist_device = torch.device("cpu") if store_on_cpu else device
    W1_hist = torch.empty((epochs,) + W1.shape, dtype=torch.float32, device=hist_device)
    W2_hist = torch.empty((epochs,) + W2.shape, dtype=torch.float32, device=hist_device)
    W3_hist = torch.empty((epochs,) + W3.shape, dtype=torch.float32, device=hist_device)
    W4_hist = torch.empty((epochs,) + W4.shape, dtype=torch.float32, device=hist_device)

    for epoch in range(epochs):
        # Forward
        Z1 = X @ W1
        A1 = sigmoid(Z1)

        Z2 = A1 @ W2
        A2 = sigmoid(Z2)

        Z3 = A2 @ W3
        A3 = sigmoid(Z3)

        Z4 = A3 @ W4
        Y_pred = sigmoid(Z4)

        # BCE (manual)
        loss = -torch.mean(
            Y * torch.log(Y_pred + 1e-8) + (1 - Y) * torch.log(1 - Y_pred + 1e-8)
        )

        # Backward
        loss.backward()

        # Update
        with torch.no_grad():
            W1 -= lr * W1.grad
            W2 -= lr * W2.grad
            W3 -= lr * W3.grad
            W4 -= lr * W4.grad

            # zero grads
            W1.grad.zero_()
            W2.grad.zero_()
            W3.grad.zero_()
            W4.grad.zero_()

            # HW02: store snapshots AFTER update (one frame per epoch) ---
            if store_on_cpu:
                W1_hist[epoch] = W1.detach().clone().cpu()
                W2_hist[epoch] = W2.detach().clone().cpu()
                W3_hist[epoch] = W3.detach().clone().cpu()
                W4_hist[epoch] = W4.detach().clone().cpu()
            else:
                W1_hist[epoch] = W1.detach().clone()
                W2_hist[epoch] = W2.detach().clone()
                W3_hist[epoch] = W3.detach().clone()
                W4_hist[epoch] = W4.detach().clone()

        loss_history.append(loss.item())

        if (epoch + 1) % 1000 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

    print("Training completed.")

    # HW02 return: 3D histories + loss
    return W1_hist, W2_hist, W3_hist, W4_hist, loss_history
