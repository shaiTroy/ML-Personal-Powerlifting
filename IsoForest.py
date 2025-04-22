import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def Iso(Inputs, Outputs):
    arr = np.array(Outputs)
    Inputs = np.array(Inputs)
    Outputs = np.array(Outputs)

    Squat = np.delete(arr, [1, 2], axis=1)
    Bench = np.delete(arr, [0, 2], axis=1)
    Deadlift = np.delete(arr, [0, 1], axis=1)
    Squat = StandardScaler().fit_transform(Squat) 
    Bench = StandardScaler().fit_transform(Bench) 
    Deadlift = StandardScaler().fit_transform(Deadlift) 
    
    scores_list = []

    for lift, lift_name in zip([Squat, Bench, Deadlift], ['Squat', 'Bench', 'Deadlift']):
        model = IsolationForest(n_estimators=200, max_samples=256, random_state=42)
        model.fit(lift)
        scores = model.decision_function(lift)
        scores_list.append(scores)
        outliers = np.sum(scores < 0)
        total_points = len(scores)
        outlier_percentage = ((total_points - outliers) / total_points) * 100
        print(f"Outlier Percentage of {lift_name}: {outlier_percentage:.2f}%")

    scores_all = np.vstack(scores_list).T  
    valid_mask = np.all(scores_all >= 0, axis=1)

    Inputs = Inputs[valid_mask]
    Outputs = Outputs[valid_mask]
    
    return Inputs, Outputs

