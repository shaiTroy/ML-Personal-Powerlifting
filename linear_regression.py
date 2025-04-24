import numpy as np
import matplotlib.pyplot as plt


#load the data
Inputs_Train = np.load('Inputs_Train.npy').astype(float)
Outputs_Train = np.load('Outputs_Train.npy').astype(float)
Inputs_Test = np.load('Inputs_Test.npy').astype(float)
Outputs_Test = np.load('Outputs_Test.npy').astype(float)


# Linear model with direct solution:
W = np.linalg.inv(Inputs_Train.T @ Inputs_Train) @ Inputs_Train.T @ Outputs_Train

predictions = Inputs_Test @ W

print(predictions)

maes =[]
lift_names = ['Squat', 'Bench', 'Deadlift']
for i, lift in enumerate(lift_names):
    mae = np.mean(np.abs(Outputs_Test[:, i] - predictions[:, i]))
    print(f"{lift} MAE: {mae:.2f}")
    maes.append(mae)
    
    sorted_indices = np.argsort(Inputs_Test[:, i])
    x_sorted = Inputs_Test[sorted_indices, i]
    y_actual_sorted = Outputs_Test[sorted_indices, i]
    y_pred_sorted = predictions[sorted_indices, i]

    plt.figure(figsize=(6, 4))
    plt.scatter(x_sorted, y_actual_sorted, color='blue', label='Actual', alpha=0.6)
    plt.plot(x_sorted, y_pred_sorted, color='red', label='Predicted Line')
    plt.title(f'{lift} - Linear Regression')
    plt.xlabel('Input Feature')
    plt.ylabel('Weight')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    
overall_mae = np.sum(maes)
print(f"Overall MAE (sum): {overall_mae:.2f}")