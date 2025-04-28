from standardize import Standard
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import os
from tqdm import tqdm


os.system('clear')

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

Inputs_Train = np.load('X_train.npy').astype(float)
Outputs_Train = np.load('squat_train.npy').astype(float)
Inputs_Test = np.load('X_test.npy').astype(float)
Outputs_Test = np.load('squat_test.npy').astype(float)

Inputs_Train = torch.tensor(Inputs_Train, dtype=torch.float32)
Outputs_Train = torch.tensor(Outputs_Train, dtype=torch.float32)
Inputs_Test = torch.tensor(Inputs_Test, dtype=torch.float32)
Outputs_Test = torch.tensor(Outputs_Test, dtype=torch.float32)

standardized = Standard(Inputs_Train.numpy(), Inputs_Test)
Inputs_Train = torch.tensor(standardized.Inputs_Train, dtype=torch.float32)
Inputs_Test = standardized.Inputs_Test.clone().detach().to(torch.float32)

Inputs_Train = Inputs_Train.to(device)
Outputs_Train = Outputs_Train.to(device).unsqueeze(1)
Inputs_Test = Inputs_Test.to(device)
Outputs_Test = Outputs_Test.to(device).unsqueeze(1)

print(Outputs_Train.shape)
print(Outputs_Test.shape)

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
        net.append(torch.nn.Linear(prev, 1))
        self.model = torch.nn.Sequential(*net)

    def forward(self, x):
        return self.model(x) 

model = MLP(
        input_dim=Inputs_Train.shape[1],
        layers=[480, 320, 192, 32],
        activation="relu",
        use_batchnorm=True,  
        dropout=0.4687129863799556              
    ).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005161277977810748, weight_decay=2.4890872386040327e-05)
criterion = torch.nn.HuberLoss()

model_save_path = "squat_model.pth"
if not os.path.exists(model_save_path):
    best_loss = float('inf')
    counter = 0
    best_model_state = None
    best_epoch = -1

    train_dataset = TensorDataset(Inputs_Train, Outputs_Train)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
        
    for epoch in range(300):

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
            if counter >= 8:
                print(f"Early stopping at epoch {epoch+1}")
                break
            
        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}/{300}, Loss: {(running_loss / len(train_loader.dataset)):.4f}")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Model reverted to best epoch: {best_epoch} (loss: {best_loss:.4f})")

    torch.save(best_model_state, model_save_path)
    print(f"Model saved to {model_save_path}")
else:
    model.load_state_dict(torch.load(model_save_path, weights_only=True))
    print(f"Model loaded from {model_save_path}")

model.eval()
y_pred1 = model(Inputs_Test.to(device))

mse = criterion(y_pred1, Outputs_Test.to(device))

print(f"Huber for Squat: {mse:.4f}")

mae_loss = torch.nn.L1Loss()
mae = mae_loss(y_pred1, Outputs_Test.to(device))

print(f"MAE for Squat: {mae:.4f}")

