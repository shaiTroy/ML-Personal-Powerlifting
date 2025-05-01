import torch
import numpy as np
from standardize import Standard
import os


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
    
def custom_huber(preds, targets, threshold=1.25):
    diffs = torch.abs(preds - targets)
    loss = torch.where(diffs < threshold, 
                       torch.zeros_like(diffs), 
                       diffs)
    return loss.mean()

def custom_loss(y_true1, y_pred1, y_true2, y_pred2):
    mse1 = custom_huber(y_pred1, y_true1)
    mse2 = custom_huber(y_pred2, y_true2)
    return mse1 + ((0.1 * (mse1 - mse2)).pow(2))
    #This function is required for proper regularization
    
def get_data(train_file, test_file = None):
    if test_file is None:
        test_file = train_file
    Inputs_Train = np.load('data/X_train.npy').astype(float)
    Outputs_Train = np.load(f'data/{train_file}').astype(float)
    Inputs_Test = np.load('data/X_test.npy').astype(float)
    Outputs_Test = np.load(f'data/{test_file}').astype(float)


    Inputs_Train = torch.tensor(Inputs_Train, dtype=torch.float32)
    Outputs_Train = torch.tensor(Outputs_Train, dtype=torch.float32)
    Inputs_Test = torch.tensor(Inputs_Test, dtype=torch.float32)
    Outputs_Test = torch.tensor(Outputs_Test, dtype=torch.float32)

    standardized = Standard(Inputs_Train.numpy(), Inputs_Test)
    Inputs_Train = torch.tensor(standardized.Inputs_Train, dtype=torch.float32)
    Inputs_Test = standardized.Inputs_Test.clone().detach().to(torch.float32)

    Inputs_Train = Inputs_Train
    Outputs_Train = Outputs_Train.unsqueeze(1)
    Inputs_Test = Inputs_Test
    Outputs_Test = Outputs_Test.unsqueeze(1)
    return Inputs_Train, Outputs_Train, Inputs_Test, Outputs_Test

def model_available(model_save_path):
    model_save_path = os.path.join('models', model_save_path)
    if not os.path.exists(f'{model_save_path}'):
        return False
    return True

def save_model_to_file(model, model_save_path):
    model_save_path = os.path.join('models', model_save_path)
    torch.save(model.state_dict(), f'{model_save_path}')
    print(f"Model saved to {model_save_path}")
    
def load_model(model, model_save_path):
    model_save_path = os.path.join('models', model_save_path)
    model.load_state_dict(torch.load(model_save_path, weights_only=True))  
    print(f"Model loaded from {model_save_path}")
    return model

    
    
    
