import torch
import torch.nn as nn
import torch.optim as optim

class LogisticRegressionPyTorch(nn.Module):

    def __init__(self, input_dim, output_dim, lr=0.01, momentum=0.9):
        super(LogisticRegressionPyTorch, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=lr, momentum=momentum)

    def forward(self, x):
        return self.linear(x)

    def train_model(self, X_train, y_train, epochs=1000, verbose=True):
        losses = []

        for epoch in range(epochs):
            self.train()
            self.optimizer.zero_grad()

            outputs = self.forward(X_train)
            loss = self.criterion(outputs, y_train)
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())

            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

        return losses

    def predict(self, X):
        self.eval()
        with torch.no_grad():
            logits = self.forward(X)
            prob = torch.softmax(logits, dim=1)
            _, predicted = torch.max(prob, 1)
        return predicted, prob

