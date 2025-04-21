import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def Iso(Inputs, Outputs):
    arr = np.array(Outputs)

    Squat = np.delete(arr, [1, 2], axis=1)
    Bench = np.delete(arr, [0, 2], axis=1)
    Deadlift = np.delete(arr, [0, 1], axis=1)
    Squat = StandardScaler().fit_transform(Squat) 
    Bench = StandardScaler().fit_transform(Bench) 
    Deadlift = StandardScaler().fit_transform(Deadlift) 
    
    for lift in [Squat, Bench, Deadlift]:

        model = IsolationForest(n_estimators=200, max_samples=256, random_state=42)
        model.fit(lift)
        scores = model.decision_function(lift)
        
        #appendthe scores to the inputs
        Inputs = np.array(Inputs)
        Inputs = np.append(Inputs, scores.reshape(-1, 1), axis=1)
        outliers = np.sum(scores < 0)  # Count outliers
        total_points = len(scores)
        outlier_percentage = ((total_points - outliers) / total_points) * 100
        lift_name = "Squat" if np.array_equal(lift, Squat) else "Bench" if np.array_equal(lift, Bench) else "Deadlift"
        print(f"Outlier Percentage of {lift_name}: {outlier_percentage:.2f}%")
        
    Outputs = np.array(Outputs)
        
    return Inputs, Outputs

