# v1 neural-network regressor, used only by the legacy phylogenetic CV (evaluate.legacy_pcv_models)
# Copied from s r c/models/nn_models.py; AttentionRegressor (Table S2 only) left there.
# Written by Michael Hoffert

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class MLPRegressor(nn.Module):
    def __init__(self, input_size=768, hidden_size=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x):
        return self.network(x)

    def predict(self, X_input):
        '''X_input: array or tensor (N, input_size) -> numpy array (N, 1)'''
        device = next(self.network.parameters()).device
        self.network.eval()
        if not isinstance(X_input, torch.Tensor):
            X_input = torch.tensor(X_input, dtype=torch.float32)
        with torch.no_grad():
            return self.network(X_input.to(device)).cpu().numpy()


def train_model(X_train, y_train, regressor_class, epochs=50, batch_size=32, lr=1e-3):
    '''Train regressor_class with Adam and MSE loss; unseeded, as in v1.'''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                            torch.tensor(y_train, dtype=torch.float32).view(-1, 1))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = regressor_class(input_size=X_train.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            loss = criterion(model(batch_x), batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(loader):.4f}")
    return model
