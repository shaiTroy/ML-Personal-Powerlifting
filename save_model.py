import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import os
from tqdm import tqdm
from NN_model import *

def save_model(train_file, model_save_path, layers, activation, dropout, learning_rate, weight_decay, batch_size, patience, use_batchnorm = True, test = False):
    os.system('clear')
    
    Inputs_Train, Outputs_Train, Inputs_Test, Outputs_Test = get_data(train_file)

    print(Outputs_Train.shape)
    print(Outputs_Test.shape)

    print(f"Training on {Inputs_Train.shape} entries")
    print(f"Testing on {Inputs_Test.shape} entries")
    
    model = MLP(
        input_dim=Inputs_Train.shape[1],
        layers=layers,
        activation=activation,
        use_batchnorm=use_batchnorm,  
        dropout=dropout              
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = custom_huber
    
    if not model_available(model_save_path):
        best_loss = float('inf')
        counter = 0
        best_model_state = None
        best_epoch = -1

        train_dataset = TensorDataset(Inputs_Train, Outputs_Train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
        for epoch in range(300):

            model.train()
            running_loss = 0.0
            for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
                xb, yb = xb, yb
                
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

        save_model(model, model_save_path)
        
    else:
        model = load_model(model, model_save_path)

    model.eval()
    y_pred1 = model(Inputs_Test)

    mse = criterion(y_pred1, Outputs_Test)

    print(f"Huber loss for Bench: {mse:.4f}")

    mae_loss = torch.nn.L1Loss()
    mae = mae_loss(y_pred1, Outputs_Test)

    print(f"MAE for Bench: {mae:.4f}")
    
    if test:        
        def accuracy(outputs, targets):
            outputs = outputs.cpu().detach().numpy().flatten()
            targets = targets.cpu().detach().numpy().flatten()
            outputs = np.round(outputs)
            targets = np.round(targets)
            correct = np.sum(np.abs(outputs - targets) < 1.25)  # Check if within 2.5kg
            print(f"Correct: {correct}, Total: {len(targets)}")
            total = len(targets)
            return correct / total
        
        outputs = model(Inputs_Test)

        print(f"Accuracy: {accuracy(outputs, Outputs_Test)*100:.2f}%")

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

        bins = np.arange(-15, 40, 5)
        hist, _ = np.histogram(Outputs_Train.cpu().detach().numpy(), bins=bins)
        print("Histogram of training outputs:")
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
        filtered_inputs = torch.tensor(filtered_inputs, dtype=torch.float32)
        filtered_outputs = torch.tensor(filtered_outputs, dtype=torch.float32).unsqueeze(1)

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




                
                
                
                
                







