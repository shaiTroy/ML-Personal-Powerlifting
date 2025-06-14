import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import os
from tqdm import tqdm
from NN_model import *

def save_model(train_file, test_file, model_save_path, layers, activation, dropout, learning_rate, weight_decay, batch_size, patience, use_batchnorm = True, test = False):
    
    Inputs_Train, Outputs_Train, Inputs_Test, Outputs_Test = get_data(train_file, test_file)

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

        save_model_to_file(model, model_save_path)
        
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
            outputs = np.round(outputs, decimals=2)
            targets = np.round(targets, decimals=2)
            correct = np.sum(np.abs(outputs - targets) < 1.25)  # Check if within 2.5kg
            print(f"Correct: {correct}, Total: {len(targets)}")
            total = len(targets)
            return correct / total
        
        outputs = model(Inputs_Test)

        print(f"Accuracy: {accuracy(outputs, Outputs_Test)*100:.2f}%")

        #create bins between -15 and 35, in range of 5. count the number of entries in each bin
        def print_histogram_percentages(data, label):
            bins = np.arange(-15, 40, 5)
            hist, _ = np.histogram(data.cpu().detach().numpy(), bins=bins)
            total = hist.sum()
            print(f"\nPercentage histogram of {label}:")
            for i in range(len(bins) - 1):
                bin_range = f"{bins[i]} to {bins[i+1]}"
                percent = (hist[i] / total) * 100 if total > 0 else 0
                print(f"Bin {bin_range}: {percent:.2f}%")

        print_histogram_percentages(outputs, "model outputs")
        print_histogram_percentages(Outputs_Test, "actual test outputs")

        def print_histogram_accuracy(model_outputs, actual_test_outputs, label):
            '''
            Here, i want to check the accuracy of the model in each histogram bin
            it should seperate the model_outputs into bins of 5, 
            and then compare check the accuracy of each data point in that bin to the nearest 2.5kg
            '''
            bins = np.arange(-15, 40, 5)
            hist, _ = np.histogram(model_outputs.cpu().detach().numpy(), bins=bins)
            print(f"\nAccuracy histogram of {label}:")
            for i in range(len(bins) - 1):
                bin_range = f"{bins[i]} to {bins[i+1]}"
                bin_mask = (model_outputs >= bins[i]) & (model_outputs < bins[i+1])
                bin_outputs = model_outputs[bin_mask]
                bin_targets = actual_test_outputs[bin_mask]
                
                if len(bin_outputs) > 0:
                    acc = accuracy(bin_outputs, bin_targets)
                    print(f"Bin {bin_range}: {acc*100:.2f}%")
                else:
                    print(f"Bin {bin_range}: No data points")
            
            print()
        print_histogram_accuracy(outputs, Outputs_Test, "model outputs")
            





                
                
                
                
                







