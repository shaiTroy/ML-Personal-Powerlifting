from standardize import Standard
import numpy as np
import torch
import os

os.system('clear')

#testing the dataset on my actual competitions
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

Inputs_Train = np.load('X_train.npy').astype(float)
Outputs_Train = np.load('deadlift_train.npy').astype(float)
Inputs_Test = np.load('X_test.npy').astype(float)
Outputs_Test = np.load('deadlift_test.npy').astype(float)

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
        layers=[32, 32],
        activation="relu",
        use_batchnorm=True,  
        dropout=0.11635918108480109              
    ).to(device)

model.load_state_dict(torch.load("deadlift_model.pth", weights_only=True))
model.eval()

n = 0
outputs = model(Inputs_Test) 

# Print the mse
mse = torch.nn.MSELoss()
print(f"MSE: {mse(outputs, Outputs_Test).item():.4f}")

#print the mae
mae = torch.nn.L1Loss()
print(f"MAE: {mae(outputs, Outputs_Test).item():.4f}")

#create bins between -15 and 35, in range of 5. count the number of entries in each bin
bins = np.arange(-15, 40, 5)
hist, _ = np.histogram(outputs.cpu().detach().numpy(), bins=bins)
print("Histogram of outputs:")
print(hist)
print("Bin edges:")
print(bins)


#create bins for the actual outputs
bins = np.arange(-15, 40, 5)
hist, _ = np.histogram(Outputs_Test.cpu().detach().numpy(), bins=bins)
print("Histogram of actual outputs:")
print(hist)
print("Bin edges:") 
print(bins)

# Create a new dataset comprised of outputs between -5 and 20
mask = (outputs.cpu().detach().numpy().flatten() >= 0) & (outputs.cpu().detach().numpy().flatten() <= 10)
print(f"Mask shape: {mask.shape}, Mask values: {np.unique(mask)}")
filtered_outputs = outputs[mask]
filtered_outputs = filtered_outputs.cpu().detach().numpy()
filtered_outputs = filtered_outputs.flatten()

# Test the model on the filtered outputs
filtered_inputs = Inputs_Test[mask, :]  # Ensure mask is 1D
print(f"Filtered inputs shape: {filtered_inputs.shape}")
print(f"Filtered inputs values: {filtered_inputs}")

filtered_inputs = filtered_inputs.cpu().detach().numpy()
filtered_inputs = torch.tensor(filtered_inputs, dtype=torch.float32).to(device)
filtered_outputs = torch.tensor(filtered_outputs, dtype=torch.float32).to(device).unsqueeze(1)

# Pass the filtered inputs through the model
filtered_outputs_pred = model(filtered_inputs)

print(f"Filtered Outputs: {filtered_outputs.shape}")
print(f"Filtered Outputs Predicted: {filtered_outputs_pred.shape}")
print(f"Model Predictions: {filtered_outputs_pred.cpu().detach().numpy().shape}")

# Print the mse
mse = torch.nn.MSELoss()
print(f"MSE: {mse(filtered_outputs_pred, filtered_outputs).item():.4f}")

#print the mae
mae = torch.nn.L1Loss()
print(f"MAE: {mae(filtered_outputs_pred, filtered_outputs).item():.4f}")



comp1 = np.array([1, 85, 545, 141, 22, 165, 177.5, 187.5, 187.5, 105, 112.5, 117.5, 117.5, 200, 220, 240, 240, 74.54, -4.4])
comp2 = np.array([1, 80.6, 557.5, 246, 22, 185, 195, 205, 205, 115, 122.5, 125, 125, 215, 227.5, -247.5, 227.5, 78.30, 1.4])
comp3 = np.array([1, 82, 592.5, 185, 23, 200, 207.5, 215, 215, 120, 125, 127.5, 127.5, 225, 235, 250, 250, 82.51, 0.5])

standardized_comp1 = standardized.standardize(comp1)
standardized_comp2 = standardized.standardize(comp2)
standardized_comp3 = standardized.standardize(comp3)

pred1 = model(torch.tensor(standardized_comp1, dtype=torch.float32).unsqueeze(0).to(device))
pred2 = model(torch.tensor(standardized_comp2, dtype=torch.float32).unsqueeze(0).to(device))
pred3 = model(torch.tensor(standardized_comp3, dtype=torch.float32).unsqueeze(0).to(device))

print(f"Pred1: {pred1.item():.4f}, actual: -12.5")
print(f"Pred2: {pred2.item():.4f}, actual: 22.5")
print(f"Pred3: {pred3.item():.4f}, actual: Not Happened")







