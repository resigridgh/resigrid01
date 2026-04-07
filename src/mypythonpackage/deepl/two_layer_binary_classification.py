import torch


def binary_cross_entropy_loss(y_pred, y_true):
    epsilon = 1e-7  # Prevent log(0) or log(1)
    y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon)
    loss = -torch.mean(
        y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred)
    )
    return loss


def binary_classification(d, n, epochs=10000, eta=0.001):
    """
    Binary Classification with Linear and Nonlinear Layers

    Returns
    -------
    [
        train_losses,   # shape: (epochs,)
        weights_W1,     # shape: (epochs, d, 48)
        weights_W2,     # shape: (epochs, 48, 16)
        weights_W3,     # shape: (epochs, 16, 32)
        weights_W4,     # shape: (epochs, 32, 1)
        W1,             # final W1
        W2,             # final W2
        W3,             # final W3
        W4              # final W4
    ]
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Synthetic dataset generation
    X = torch.randn(n, d, dtype=torch.float32, device=device)
    Y = (X.sum(axis=1, keepdim=True) > 2).float()

    current_dtype = torch.float32

    W1 = (
        torch.randn(d, 48, device=device, dtype=current_dtype)
        * torch.sqrt(torch.tensor(1.0 / d, device=device, dtype=current_dtype))
    ).requires_grad_(True)

    W2 = (
        torch.randn(48, 16, device=device, dtype=current_dtype)
        * torch.sqrt(torch.tensor(1.0 / 48, device=device, dtype=current_dtype))
    ).requires_grad_(True)

    W3 = (
        torch.randn(16, 32, device=device, dtype=current_dtype)
        * torch.sqrt(torch.tensor(1.0 / 16, device=device, dtype=current_dtype))
    ).requires_grad_(True)

    W4 = (
        torch.randn(32, 1, device=device, dtype=current_dtype)
        * torch.sqrt(torch.tensor(1.0 / 32, device=device, dtype=current_dtype))
    ).requires_grad_(True)

    train_losses = torch.zeros(epochs, device=device)

    # Store intermediate weights from every epoch
    weights_W1 = torch.zeros(epochs, d, 48, dtype=current_dtype, device=device)
    weights_W2 = torch.zeros(epochs, 48, 16, dtype=current_dtype, device=device)
    weights_W3 = torch.zeros(epochs, 16, 32, dtype=current_dtype, device=device)
    weights_W4 = torch.zeros(epochs, 32, 1, dtype=current_dtype, device=device)

    for epoch in range(epochs):
        Z1 = torch.matmul(X, W1)
        Z1 = torch.matmul(Z1, W2)
        A1 = 1 / (1 + torch.exp(-Z1))

        Z2 = torch.matmul(A1, W3)
        Z2 = torch.matmul(Z2, W4)
        A2 = 1 / (1 + torch.exp(-Z2))

        YPred = A2

        train_loss = binary_cross_entropy_loss(YPred, Y)
        train_loss.backward()

        with torch.no_grad():
            W1 -= eta * W1.grad
            W2 -= eta * W2.grad
            W3 -= eta * W3.grad
            W4 -= eta * W4.grad

            # Store a copy of weights after update at this epoch
            weights_W1[epoch] = W1.clone()
            weights_W2[epoch] = W2.clone()
            weights_W3[epoch] = W3.clone()
            weights_W4[epoch] = W4.clone()

            # Store loss
            train_losses[epoch] = train_loss

            # Zero gradients
            W1.grad.zero_()
            W2.grad.zero_()
            W3.grad.zero_()
            W4.grad.zero_()

        if epoch % 100 == 0:
            print(f"Epoch {epoch} loss: {train_loss.item():.4f}")

    return [
        train_losses,
        weights_W1,
        weights_W2,
        weights_W3,
        weights_W4,
        W1,
        W2,
        W3,
        W4,
    ]
