import csv
from datetime import datetime
import numpy as np
import sys
from sklearn.model_selection import train_test_split
import os
from IsoForest import Iso

os.system('clear')

def importData():
    with open('openipf-2025-04-26-3ab5a183.csv', mode='r') as file:
        reader = csv.reader(file)
        data = []
        for row in reader:
            currentRow = []
            if row[2] != 'SBD' or row[3] != 'Raw' or row[7] == 'Special Olympics' or row[30] == '': continue
            if row[7] == 'Juniors': row[7] = 'Junior'
            if row[7] == 'Sub-Juniors': row[7] = 'Sub-Junior'
            for i in [0, 1, 8, 25, 36, 4, 5, 6, 10, 11, 12, 14, 15, 16, 17, 19, 20, 21, 22, 24, 30]:
                    currentRow.append(row[i])
            data.append(currentRow)
        return data
    #row format:
    # [0] = Name
    # [1] = Sex
    # [2] = BodyWeight
    # [3] = Total
    # [4] = Date
    # [5] = Age
    # [6] = AgeClass
    # [7] = BirthYear
    # [8] = Squat1
    # [9] = Squat2
    # [10] = Squat3
    # [11] = Best Squat
    # [12] = Bench1
    # [13] = Bench2
    # [14] = Bench3
    # [15] = Best Bench
    # [16] = Deadlift1
    # [17] = Deadlift2
    # [18] = Deadlift3
    # [19] = Best Deadlift
    # [20] = GL
    
def sortData(data):
    # Sort by date at index 4
    data.sort(key=lambda x: datetime.strptime(x[4], '%Y-%m-%d'))
    # Sort by first and last name at index 0
    data.sort(key=lambda x: (x[0].split()[0], x[0].split()[1]) if len(x[0].split()) > 1 else (x[0].split()[0], ''))
    return data

def updateDates(filtered):
    first_occurrence_date = None
    current_name = None

    for entry in filtered:
        name = entry[0]
        date_str = entry[4]

        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            print(f"Skipping entry due to incorrect date format: {date_str}")
            continue
        
        if name != current_name:
            current_name = name
            first_occurrence_date = date_obj
            entry[4] = 0
        else:
            days_difference = (date_obj - first_occurrence_date).days
            entry[4] = days_difference
    i = 0
    n = len(filtered)
    while i < len(filtered)-1:
        if filtered[i][4] == filtered[i+1][4]:
            filtered.pop(i)
        else:
            i += 1 
    if filtered[-1][4] == 0:
        filtered.pop(-1)
    print(f"Filtered out {n - len(filtered)} entries with the same lifter and date, {len(filtered)} entries remaining.")

    i = 0
    n = len(filtered)
    while i < len(filtered)-1:
        try:
            filtered[i][5] = float(filtered[i][5])
        except:
            try: 
                filtered[i][5] = float(filtered[i][6][:2])
            except:
                try:
                    filtered[i][5] = float(filtered[i][7][:2])
                except:
                    filtered.pop(i)
                    continue
        for j in range(8, 21):
            try:
                filtered[i][j] = float(filtered[i][j])
            except:
                filtered.pop(i)
                i -= 1
                break
            if filtered[i][j] == 0:
                print(f"Found 0 attempt in {filtered[i]}")
                filtered.pop(i)
                i -= 1
                break

        i += 1
    print(f"Filtered out {n - len(filtered)} entries with no age or incorrect attempts, {len(filtered)} entries remaining.")
    
    return filtered

def transform_and_save_arrays(filtered):
    comps = []
    cur_name = None
    holder = []


    for row in filtered:
        if cur_name != row[0]:
            cur_name = row[0]
            holder = [row[1], float(row[2]), float(row[3]), float(row[4]), row[5], float(row[8]), float(row[9]), float(row[10]), float(row[11]), float(row[12]), float(row[13]), float(row[14]), float(row[15]), float(row[16]), float(row[17]), float(row[18]), float(row[19]), row[20]]
        else:
            if holder[0] == 'M':
                holder[0] = 1
            else:
                holder[0] = 0
            holder[3] = row[4] - holder[3]
            holder.append(float(row[2])-holder[1])
            holder.append(float(row[3])-holder[2])
            holder.append(float(row[11])-holder[8]) #append change in squat
            holder.append(float(row[15])-holder[12]) #append change in bench
            holder.append(float(row[19])-holder[16]) #append change in deadlift
            comps.append(holder)
            holder = [row[1], float(row[2]), float(row[3]), float(row[4]), row[5], float(row[8]), float(row[9]), float(row[10]), float(row[11]), float(row[12]), float(row[13]), float(row[14]), float(row[15]), float(row[16]), float(row[17]), float(row[18]), float(row[19]), row[20]]
    
    print(f"Number of entries: {len(comps)}")

    #Send to unsupervised algo to learn injured vs healthy
    #Inputs, extra = np.array(comps)
    Inputs, extra = Iso(comps)
    
    #Array looks as following:
    
    #Inputs
    #[0] = Sex (0 for F, 1 for M)
    #[1] = Init Weight
    #[2] = Init Total
    #[3] = Days between competitions
    #[4] = Age
    #[5] = Squat1
    #[6] = Squat2
    #[7] = Squat3
    #[8] = Best Squat
    #[9] = Bench1
    #[10] = Bench2
    #[11] = Bench3
    #[12] = Best Bench
    #[13] = Deadlift1
    #[14] = Deadlift2
    #[15] = Deadlift3
    #[16] = Best Deadlift
    #[17] = GL
    #[18] = Change Weight

    
    #Outputs
    #[0] = Change in Squat
    #[1] = Change in Bench
    #[2] = Change in Deadlift
    
    
    np.random.shuffle(Inputs)

    # Separate features and targets
    X = Inputs[:, :-4]         # Input features
    y = Inputs[:, -3:]         # Output columns
    Xextra = extra[:, :-4]
    yextra = extra[:, -3:]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train = np.concatenate((X_train, Xextra), axis=0)
    y_train = np.concatenate((y_train, yextra), axis=0)
    
    perm = np.random.permutation(len(X_train))
    X_train = X_train[perm]
    y_train = y_train[perm]

    # Separate the outputs
    squat_train = y_train[:, 0]
    bench_train    = y_train[:, 1]
    deadlift_train    = y_train[:, 2]

    squat_test = y_test[:, 0]
    bench_test = y_test[:, 1]
    deadlift_test = y_test[:, 2]
    
    print(f"Number of training entries: {len(X_train)}")
    print(f"Number of testing entries: {len(X_test)}")
 
    os.makedirs('data', exist_ok=True)

    # Save inputs
    np.save('data/X_train.npy', X_train)
    np.save('data/X_test.npy', X_test)

    # Save separated outputs
    np.save('data/deadlift_train.npy', deadlift_train)
    np.save('data/bench_train.npy', bench_train)
    np.save('data/squat_train.npy', squat_train)

    np.save('data/deadlift_test.npy', deadlift_test)
    np.save('data/bench_test.npy', bench_test)
    np.save('data/squat_test.npy', squat_test)
    
    
if __name__ == '__main__':
    data = sortData(importData())
    filtered = updateDates(data)
    transform_and_save_arrays(filtered)
    print("Data has been preprocessed and saved to files.")
    
    