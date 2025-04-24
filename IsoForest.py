import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def Iso(arr):
    arr = np.array(arr)

    Input = np.delete(arr, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22], axis=1)
    Input = StandardScaler().fit_transform(Input) 

    model = IsolationForest(n_estimators=200, max_samples=256, random_state=42)
    model.fit(Input)

    scores = model.decision_function(Input)

    outliers = np.sum(scores < 0)  # Count outliers
    total_points = len(scores)  # Total number of samples

    outlier_percentage = ((total_points - outliers) / total_points) * 100
    print(f"Outlier Percentage: {outlier_percentage:.2f}%, number of remaining points: {total_points - outliers}")
    
    mask = scores >= 0  

    filtered_arr = arr[mask]
    
    input = np.array((filtered_arr))

    
    '''
    plt.figure(figsize=(8, 6))
    plt.scatter(arr[:, 3], Input[:, 0], c=scores, cmap='coolwarm', s=100)
    plt.xlabel("Init Total")
    plt.ylabel("Change in Total (Standardized)")
    plt.title('Anomaly Scores for Outlier Detection')
    plt.show()
    '''
    '''
    valid_indices = scores >= 0
    valid_values = Input[valid_indices]

    if len(valid_values) > 0:
        max_value = np.max(valid_values)
        min_value = np.min(valid_values)
        print(f"Greatest number in Input with score >= 0: {max_value:.2f}")
        print(f"Smallest number in Input with score >= 0: {min_value:.2f}")
    '''
    
    return input