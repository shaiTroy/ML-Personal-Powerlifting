import numpy as np
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
import os
import matplotlib.pyplot as plt

os.system('clear')

Inputs_Train = np.load('Inputs_Train.npy').astype(float)
Outputs_Train = np.load('Outputs_Train.npy').astype(float).ravel()
Inputs_Test = np.load('Inputs_Test.npy').astype(float)
Outputs_Test = np.load('Outputs_Test.npy').astype(float)

Outputs_Train = Outputs_Train.reshape(-1, 3)


feature_names = ['Sex', 'W', 'DBC', 'Age', 'BS', 'BB', 'BD', 'GL', 'CW']
lift_names = ['Squat', 'Bench', 'Deadlift']

for i in range(3):
    print(f"\n=== {lift_names[i]} ===")
    
    # Train a separate model for each output
    model = DecisionTreeRegressor()
    model.fit(Inputs_Train, Outputs_Train[:, i])
    
    # Predict and evaluate
    y_pred = model.predict(Inputs_Test)
    mse = mean_squared_error(Outputs_Test[:, i], y_pred)
    mae = mean_absolute_error(Outputs_Test[:, i], y_pred)
    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, Tree Depth: {model.get_depth()}")

    '''
    # Plot the decision tree
    plt.figure(figsize=(16, 8))
    plot_tree(model, feature_names=feature_names, filled=True)
    plt.title(f"Decision Tree for {lift_names[i]}")
    plt.show()
    '''
    for j in range(100):
        print(f"Output: {y_pred[j]}, Target: {Outputs_Test[j][i]}")