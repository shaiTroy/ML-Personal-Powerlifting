from standardize import Standard
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import os
from tqdm import tqdm

os.system('clear')

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

Inputs_Train = np.load('Inputs_Train.npy').astype(float)
Outputs_Train = np.load('Outputs_Train.npy').astype(float)
Inputs_Test = np.load('Inputs_Test.npy').astype(float)
Outputs_Test = np.load('Outputs_Test.npy').astype(float)

Inputs_Train = torch.tensor(Inputs_Train, dtype=torch.float32)
Outputs_Train = torch.tensor(Outputs_Train, dtype=torch.float32)
Inputs_Test = torch.tensor(Inputs_Test, dtype=torch.float32)
Outputs_Test = torch.tensor(Outputs_Test, dtype=torch.float32)

standardized = Standard(Inputs_Train.numpy(), Inputs_Test)
Inputs_Train = torch.tensor(standardized.Inputs_Train, dtype=torch.float32)
Inputs_Test = standardized.Inputs_Test.clone().detach().to(torch.float32)

Inputs_Train = Inputs_Train.to(device)
Outputs_Train = Outputs_Train.to(device)
Inputs_Test = Inputs_Test.to(device)
Outputs_Test = Outputs_Test.to(device)

print(f"Training on {Inputs_Train.shape} entries")
print(f"Testing on {Inputs_Test.shape} entries")

class MLP(torch.nn.Module):
    def __init__(self, input_dim, layers, activation, use_batchnorm=False, dropout=0.0):
        super().__init__()
        activ = {
            "relu": torch.nn.ReLU(),
            "tanh": torch.nn.Tanh(),
            "logistic": torch.nn.Sigmoid()
        }[activation]

        net = []
        prev = input_dim
        for l in layers:
            net.append(torch.nn.Linear(prev, l))
            if use_batchnorm:
                net.append(torch.nn.BatchNorm1d(l))
            net.append(activ)
            if dropout > 0:
                net.append(torch.nn.Dropout(dropout))
            prev = l
        net.append(torch.nn.Linear(prev, 3))
        self.model = torch.nn.Sequential(*net)

    def forward(self, x):
        return self.model(x)

def custom_mae(y_pred, y_true):
    abs_error = (y_pred - y_true).abs()  
    column_mae = abs_error.mean(dim=0)
    return column_mae.sum()   

model = MLP(
        input_dim=Inputs_Train.shape[1],
        layers=[448, 32],
        activation="logistic",
        use_batchnorm=False,  
        dropout=0.14841265123200637              
    ).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0014632171993503094, weight_decay=1.768622834823392e-06)
criterion = custom_mae

model_save_path = "best_model.pth"
if not os.path.exists(model_save_path):
    best_loss = float('inf')
    counter = 0
    best_model_state = None
    best_epoch = -1

    train_dataset = TensorDataset(Inputs_Train, Outputs_Train)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
        
    for epoch in range(150):

        model.train()
        running_loss = 0.0
        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            xb, yb = xb.to(device), yb.to(device)
            
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)

        if running_loss < best_loss:
            best_loss = running_loss
            best_model_state = model.state_dict()
            best_epoch = epoch + 1  # Store 1-based epoch index
            counter = 0
        else:
            counter += 1
            if counter >= 6:
                print(f"Early stopping at epoch {epoch+1}")
                break
            
        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}/{150}, Loss: {(running_loss / len(train_loader.dataset)):.4f}")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Model reverted to best epoch: {best_epoch} (loss: {best_loss:.4f})")

    torch.save(best_model_state, model_save_path)
    print(f"Model saved to {model_save_path}")
else:
    model.load_state_dict(torch.load(model_save_path))
    print(f"Model loaded from {model_save_path}")

model.eval()
y_pred1 = model(Inputs_Test.to(device))

def final_custom_mae(y_pred, y_true):
    abs_error = (y_pred - y_true).abs()  
    column_mae = abs_error.mean(dim=0)
    return column_mae 

mae = final_custom_mae(y_pred1, Outputs_Test.to(device))
for i, col in enumerate(['Squat', 'Bench', 'Deadlift']):
    print(f"MAE for {col}: {mae[i].item():.4f}")
