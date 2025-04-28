from standardize import Standard
import numpy as np
import torch
import optuna
from torch.utils.data import DataLoader, TensorDataset
import os
from tqdm import tqdm
from NN_model import MLP, custom_huber, custom_loss

def optuna_running(train_file, study_name):
    device = torch.device("cpu")

    Inputs_Train = np.load('X_train.npy').astype(float)
    Outputs_Train = np.load(train_file).astype(float)
    Inputs_Test = np.load('X_test.npy').astype(float)

    val_split = int(len(Inputs_Train) * 0.8)
    Inputs_Validation = torch.tensor(Inputs_Train[val_split:], dtype=torch.float32)
    Outputs_Validation = torch.tensor(Outputs_Train[val_split:], dtype=torch.float32)
    Inputs_Train = torch.tensor(Inputs_Train[:val_split], dtype=torch.float32)
    Outputs_Train = torch.tensor(Outputs_Train[:val_split], dtype=torch.float32)

    standardized = Standard(Inputs_Train.numpy(), Inputs_Test, Inputs_Validation.numpy())
    Inputs_Train = torch.tensor(standardized.Inputs_Train, dtype=torch.float32)
    Inputs_Test = torch.tensor(standardized.Inputs_Test, dtype=torch.float32)
    Inputs_Validation = torch.tensor(standardized.Inputs_Validation, dtype=torch.float32)

    mid_idx = len(Inputs_Validation) // 2
    Inputs_Val_1 = Inputs_Validation[:mid_idx]
    Outputs_Val_1 = Outputs_Validation[:mid_idx]
    Inputs_Val_2 = Inputs_Validation[mid_idx:]
    Outputs_Val_2 = Outputs_Validation[mid_idx:]

    Inputs_Train = Inputs_Train.to(device)
    Inputs_Val_1 = Inputs_Val_1.to(device)
    Inputs_Val_2 = Inputs_Val_2.to(device)
    Outputs_Train = Outputs_Train.to(device)
    Outputs_Val_1 = Outputs_Val_1.to(device).unsqueeze(1)
    Outputs_Val_2 = Outputs_Val_2.to(device).unsqueeze(1)

    print(f"Training on {Inputs_Train.shape} entries")
    print(f"Validation on {Inputs_Validation.shape} entries, separated into {Inputs_Val_1.shape} and {Inputs_Val_2.shape}")
    print(f"Testing on {Inputs_Test.shape} entries")
        
    mse = custom_huber

    study_dir = os.path.join('/Users/shaitroy/d/ML-Personal-Powerlifting', 'optuna_studies')
    os.makedirs(study_dir, exist_ok=True)
    storage_path = f'sqlite:///{study_dir}/mlp_study.db'

    sampler = optuna.samplers.TPESampler(n_startup_trials=50)  # You can increase this number
    study = optuna.create_study(
        direction="minimize",
        storage=storage_path,
        study_name=study_name,
        sampler=sampler,
        load_if_exists=True
    )

    def objective(trial):
        n_layers = trial.suggest_int("n_layers", 2, 10)
        max_nodes = trial.suggest_int("max_nodes", 32, 512, step=32)

        step_sizes = [max_nodes - i * ((max_nodes-32) // (n_layers - 1)) for i in range(n_layers)]
        layers = [int(round(n / 32) * 32) for n in step_sizes]
        layers = [max(32, min(max_nodes, l)) for l in layers]

        learning_rate = trial.suggest_float("learning_rate_init", 1e-4, 1e-1, log=True)
        alpha = trial.suggest_float("alpha", 1e-6, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256, 512, 1024, 2048])
        activation = trial.suggest_categorical("activation", ["relu", "tanh", "logistic"])

        patience = trial.suggest_int("patience", 3, 10)  
        dropout = trial.suggest_float("dropout", 0.0, 0.5)  
        use_batchnorm = trial.suggest_categorical("use_batchnorm", [True, False])  

        model = MLP(
            input_dim=Inputs_Train.shape[1],
            layers=layers,
            activation=activation,
            use_batchnorm=use_batchnorm,  
            dropout=dropout              
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=alpha)
        criterion = mse

        # Early stopping parameters
        best_loss = float('inf')
        counter = 0
        best_model_state = None
        best_epoch = -1
        
        train_dataset = TensorDataset(Inputs_Train, Outputs_Train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        print("Trial starting training with parameters:")
        print(f"Layers: {layers}, Learning Rate: {learning_rate}, Alpha: {alpha}, Batch Size: {batch_size}, Activation: {activation}, Dropout: {dropout}, Use BatchNorm: {use_batchnorm}, Patience: {patience}")


        for epoch in range(300):

            model.train()
            running_loss = 0.0
            for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
                xb, yb = xb.to(device), yb.to(device).unsqueeze(1)
                
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
                if counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch+1}/{300}, Loss: {(running_loss / len(train_loader.dataset)):.4f}")
        
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"Model reverted to best epoch: {best_epoch} (loss: {best_loss:.4f})")

        # Validation after training ends
        model.eval()
        y_pred1 = model(Inputs_Val_1.to(device))
        y_pred2 = model(Inputs_Val_2.to(device))
        final_loss = custom_loss(Outputs_Val_1, y_pred1, Outputs_Val_2, y_pred2)

        return final_loss

    study.optimize(objective, n_trials=250)

    valid_trials = [t for t in study.trials if t.value is not None]

    top_trials = sorted(valid_trials, key=lambda t: t.value)[:5]

    def compute_layers(n_layers, max_nodes):
        step_sizes = [max_nodes - i * ((max_nodes - 32) // (n_layers - 1)) for i in range(n_layers)]
        layers = [int(round(n / 32) * 32) for n in step_sizes]
        return [max(32, min(max_nodes, l)) for l in layers]

    for i, trial in enumerate(top_trials):
        print(f"Rank {i+1}: Trial #{trial.number}")
        print(f"  Validation Loss: {trial.value}")
        n_layers = trial.params.get('n_layers')
        max_nodes = trial.params.get('max_nodes')
        if n_layers is not None and max_nodes is not None:
            layers = compute_layers(n_layers, max_nodes)
            print(f"  Layers: {layers}")
        print(f"  learning_rate_init: {trial.params.get('learning_rate_init', 'N/A')}")
        print(f"  alpha: {trial.params.get('alpha', 'N/A')}")
        print(f"  batch_size: {trial.params.get('batch_size', 'N/A')}")
        print(f"  activation: {trial.params.get('activation', 'N/A')}")
        print(f"  dropout: {trial.params.get('dropout', 'N/A')}")
        print(f"  use_batchnorm: {trial.params.get('use_batchnorm', 'N/A')}")
        print(f"  patience: {trial.params.get('patience', 'N/A')}")
        print()