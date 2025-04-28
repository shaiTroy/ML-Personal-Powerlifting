import torch


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
    
