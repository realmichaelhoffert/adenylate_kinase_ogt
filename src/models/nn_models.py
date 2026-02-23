# PLMs
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler, random_split, TensorDataset
import torch.optim as optim
from torch.optim import Adam

from torch.nn import HuberLoss, MSELoss

# ESM
import esm



class MLPRegressor(nn.Module):
    def __init__(self, input_size=768, hidden_size=256):
        super(MLPRegressor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1), # Prevents overfitting
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, x):
        return self.network(x)

    def predict(self, X_input):
        """
        X_input: A numpy array or torch tensor of shape (N, 768)
        """
        device = next(self.network.parameters()).device # Get model's current device
        self.network.eval()                             # Set to evaluation mode
        
        # Convert input to tensor if it's currently a numpy array
        if not isinstance(X_input, torch.Tensor):
            X_input = torch.tensor(X_input, dtype=torch.float32)
        
        X_input = X_input.to(device)
        
        with torch.no_grad():                    # Disable gradient tracking
            predictions = self.network(X_input)
        
        return predictions.cpu().numpy()         # Move back to CPU and convert to numpy

class AttentionRegressor(nn.Module):
    def __init__(self, input_size=768, hidden_size=256):
        super(AttentionRegressor, self).__init__()
        
        # attention:
        # This will operate on the last dimension (768) 
        # to produce a single score for each M.
        self.attention_net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # 2. Regression Head
        self.regressor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1), # Prevents overfitting
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x, return_attention=False):
        # Ensure x is (Batch, M, 768)
        if x.dim() == 2:
            x = x.unsqueeze(0) # Add batch dim if a single example is passed
            
        # scores: (Batch, M, 768) -> (Batch, M, 1)
        scores = self.attention_net(x)
        
        # weights: (Batch, M, 1) - normalized across the M dimension
        weights = F.softmax(scores, dim=1)

        # context_vector: (Batch, M, 1) * (Batch, M, 768) -> (Batch, 768)
        # We use batch matrix multiplication (bmm) or element-wise + sum
        context_vector = torch.sum(weights * x, dim=1)
        
        
        predictions = self.regressor(context_vector)
        
        if return_attention:
            return predictions, weights
        return predictions

    def predict(self, X, return_weights=False):
        device = next(self.parameters()).device
        self.eval()
        
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        
        # Ensure it's on the right device
        X = X.to(device)
        
        with torch.no_grad():
            # If X is 2D (M, 768), make it 3D (1, M, 768)
            if X.dim() == 2:
                X = X.unsqueeze(0)
                
            if return_weights:
                preds, weights = self.forward(X, return_attention=True)
                return preds.cpu().numpy(), weights.cpu().numpy()
            else:
                preds = self.forward(X)
                return preds.cpu().numpy()

def train_model(X_train, y_train, regressor_class, epochs=50, batch_size=32, lr=1e-3):
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Prepare data
    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), 
                            torch.tensor(y_train, dtype=torch.float32).view(-1, 1))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model, loss, and optimizer
    model = regressor_class(input_size=768).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # Forward pass
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(loader):.4f}")
            
    return model