from standardize import Standard
import numpy as np
import torch

Inputs_Train = np.load('X_train.npy').astype(float)
Inputs_Test = np.load('X_test.npy').astype(float)

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
    
class CombinedModel:
    def __init__(self):
        self.Squat_model = MLP(
            input_dim=19,
            layers=[160, 32],
            activation="relu",
            use_batchnorm=True,  
            dropout=0.15119733872980617              
        )
        self.Squat_model.load_state_dict(torch.load('Squat_model.pth', weights_only=True))
        self.Squat_model.eval()
        
        self.bench_model = MLP(
            input_dim=19,
            layers=[160, 32],
            activation="relu",
            use_batchnorm=False,  
            dropout=0.11635918108480109              
        )
        self.bench_model.load_state_dict(torch.load('Bench_model.pth', weights_only=True))
        self.bench_model.eval()
        
        self.deadlift_model = MLP(
            input_dim=19,
            layers=[32, 32],
            activation="relu",
            use_batchnorm=True,  
            dropout=0.11635918108480109              
        )
        self.deadlift_model.load_state_dict(torch.load('Deadlift_model.pth', weights_only=True))
        self.deadlift_model.eval()
            
    def predict(self, input_data):
        def round_to_nearest_2_5(value):
            return round(value * 2 / 5) * 2.5
        copy = input_data.copy()
        Standardized = Standard(Inputs_Train, Inputs_Test)
        input_data = torch.tensor(Standardized.standardize(input_data), dtype=torch.float32)
        result = []
        for i in range(len(input_data)):
            input_batch = input_data[i].unsqueeze(0)  # Add batch dimension
            squat = round_to_nearest_2_5(self.Squat_model(input_batch).squeeze(0).item() + copy[i][8].item())
            bench = round_to_nearest_2_5(self.bench_model(input_batch).squeeze(0).item() + copy[i][12].item())
            deadlift = round_to_nearest_2_5(self.deadlift_model(input_batch).squeeze(0).item() + copy[i][16].item())
            result.append([squat, bench, deadlift])
        return result
        
#Shai Comps
comp1 = np.array([1, 85, 545, 141, 21, 165, 177.5, 187.5, 187.5, 105, 112.5, 117.5, 117.5, 200, 220, 240, 240, 74.54, -4.4])
comp2 = np.array([1, 80.6, 557.5, 246, 21, 185, 195, 205, 205, 115, 122.5, 125, 125, 215, 227.5, -247.5, 227.5, 78.30, 1.4])
comp3 = np.array([1, 82, 592.5, 185, 22, 400, 207.5, 215, 215, 120, 125, 127.5, 127.5, 225, 235, 250, 250, 82.51, 0.4])

#Jan Comps
comp4 = np.array([1, 91.8, 542.5, 386, 19, 185, 195, 200, 200, 95, 102.5, -115, 102.5, 225, 240, -245, 240, 71.40, 0.8])
comp5 = np.array([1, 92.6, 557.5, 139, 20, 187.5, 197.5, 205, 205, 92.5, 100, 110, 110, 237.5, 250, 270, 270, 76.68, -1])

#Michael Comps
comp6 = np.array([1, 71.9, 532.5, 197, 19, 175, 180, 190, 190, 115, 120, 127.5, 127.5, 185, 205, 215, 215, 79.36, 1.5])
comp7 = np.array([1, 73.5, 580, 533, 20, 185, 195, 205, 205, 125, 132.5, -142.5, 132.5, 215, -227.5, 242.5, 242.5, 85.51, 0.4])
comp8 = np.array([1, 73.8, 615, 185, 22, 200, 210, 212.5, 212.5, 140, -147.5, -147.5, 140, 235, 247.5, 262.5, 262.5, 90.40, -0.6])

comps = np.array([comp1, comp2, comp3, comp4, comp5, comp6, comp7, comp8, ])
model = CombinedModel()
predictions = model.predict(comps)
print("Shai Comps")
print(f"Predection for 2nd comp: {predictions[0]}, actual: [205, 125, 227.5]")
print(f"Predection for 3rd comp: {predictions[1]}, actual: [215, 127.5, 250]")
print(f"Predection for 4th comp: {predictions[2]}")
print("Jan Comps")
print(f"Predection for 2nd comp: {predictions[3]}, actual: [205, 110, 270]")
print(f"Predection for 3rd comp: {predictions[4]}")
print("Michael Comps")
print(f"Predection for 2nd comp: {predictions[5]}, actual: [185, 132.5, 242.5]")
print(f"Predection for 3rd comp: {predictions[6]}, actual: [212.5, 140, 262.5]")
print(f"Predection for 4th comp: {predictions[7]}")


