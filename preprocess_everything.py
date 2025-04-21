import csv
from datetime import datetime
import numpy as np
import sys
from IsoForest import Iso
import os
os.system('clear')  

def importData():
    with open('openipf-2025-04-12-49fad7a9.csv', mode='r') as file:
        reader = csv.reader(file)
        data = []
        for row in reader:
            currentRow = []
            if row[2] != 'SBD' or row[3] != 'Raw' or row[7] == 'Special Olympics' or row[30] == '': continue
            if row[7] == 'Juniors': row[7] = 'Junior'
            if row[7] == 'Sub-Juniors': row[7] = 'Sub-Junior'
            for i in [0, 1, 8, 36, 4, 5, 6, 14, 19, 24, 30]:
                    currentRow.append(row[i])
            data.append(currentRow)
        return data
    #row format:
    # [0] = Name
    # [1] = Sex
    # [2] = BodyWeight
    # [3] = Date
    # [4] = Age
    # [5] = AgeClass
    # [6] = BirthYear
    # [7] = Best Squat
    # [8] = Best Bench
    # [9] = Best Deadlift
    # [10] = GL
    
def sortData(data):
    # Sort by date at index 3
    data.sort(key=lambda x: datetime.strptime(x[3], '%Y-%m-%d'))
    # Sort by first and last name at index 0
    data.sort(key=lambda x: (x[0].split()[0], x[0].split()[1]) if len(x[0].split()) > 1 else (x[0].split()[0], ''))
    return data

def updateDates(filtered):
    first_occurrence_date = None
    current_name = None

    for entry in filtered:
        name = entry[0]
        date_str = entry[3]

        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            print(f"Skipping entry due to incorrect date format: {date_str}")
            continue
        
        if name != current_name:
            current_name = name
            first_occurrence_date = date_obj
            entry[3] = 0
        else:
            days_difference = (date_obj - first_occurrence_date).days
            entry[3] = days_difference
    i = 0
    n = len(filtered)
    while i < len(filtered)-1:
        if filtered[i][3] == filtered[i+1][3]:
            filtered.pop(i)
        else:
            i += 1 
    if filtered[-1][3] == 0:
        filtered.pop(-1)
    print(f"Filtered out {n - len(filtered)} entries with the same lifter and date, {len(filtered)} entries remaining.")
    
    #filterNamesWithAtLeastTwoOccurrences(filtered) # This is can be uncommented to check if the filter function removes 0 names

    #filter out entries with no age or incorrect attempts
    i = 0
    n = len(filtered)
    while i < len(filtered)-1:
        try:
            filtered[i][4] = float(filtered[i][4])
        except:
            try: 
                filtered[i][4] = float(filtered[i][5][:2])
            except:
                try:
                    filtered[i][4] = float(filtered[i][6][:2])
                except:
                    filtered.pop(i)
                    continue
        try:
            filtered[i][7] = float(filtered[i][7])
            filtered[i][8] = float(filtered[i][8])
            filtered[i][9] = float(filtered[i][9])
        except:
            filtered.pop(i)
            continue
        i += 1
    print(f"Filtered out {n - len(filtered)} entries with no age or incorrect attempts, {len(filtered)} entries remaining.")
    
    return filtered

def transform_and_save_arrays(filtered):
    cur_name = None
    holder = []

    Inputs = []
    Outputs = []
    
    for row in filtered:
        if cur_name != row[0]:
            cur_name = row[0]
            holder = [row[1], float(row[2]), float(row[3]), float(row[4]), float(row[7]), float(row[8]), float(row[9]), float(row[10])]
           #holder = [gender, init bw,      date,               age,            squat,      bench,          deadlift,           GL     ]
        else:
            if holder[0] == 'M':
                holder[0] = 1
            else:
                holder[0] = 0
            holder[2] = row[3] - holder[2]
            holder.append(float(row[2])-holder[1]) #add change in bw
            Inputs.append(holder) #add all inputs to array
            Outputs.append([float(row[7])-holder[4], float(row[8])-holder[5], float(row[9])-holder[6]])# add change in each total to output
            holder = [row[1], float(row[2]), float(row[3]), float(row[4]), float(row[7]), float(row[8]), float(row[9]), float(row[10])]
    
    print(f"Number of entries: {len(Inputs)}")

    #Send to unsupervised algo to learn injured vs healthy
    Inputs, Outputs= Iso(Inputs, Outputs)
    
    #Array looks as following:
    
    #Inputs
    #[0] = Sex (0 for F, 1 for M)
    #[1] = Init Weight
    #[2] = Days between competitions
    #[3] = Age
    #[4] = Best Squat
    #[5] = Best Bench
    #[6] = Best Deadlift
    #[7] = GL
    #[8] = Change Weight
    #[9] = Isolation score

    
    #Outputs
    #[0] = Change Squat
    #[1] = Change Bench
    #[2] = Change Deadlift
    
    
    #for i in range (10):
    #    print(f"Input: {Inputs[i]}, Output: {Outputs[i]}")
    
    #split up the data into training and testing splits
    np.random.shuffle(Inputs)
    np.random.shuffle(Outputs)
    
    Inputs_Train = Inputs[:int(len(Inputs)*0.8)]
    Outputs_Train = Outputs[:int(len(Outputs)*0.8)]
    Inputs_Test = Inputs[int(len(Inputs)*0.8):]
    Outputs_Test = Outputs[int(len(Outputs)*0.8):]
    
    print(f"Number of training samples: {len(Inputs_Train)} number of test samples: {len(Inputs_Test)}")

    # Save the arrays to files
    np.save('Inputs_Train.npy', Inputs_Train)
    np.save('Outputs_Train.npy', Outputs_Train)
    np.save('Inputs_Test.npy', Inputs_Test)
    np.save('Outputs_Test.npy', Outputs_Test)
    
if __name__ == '__main__':
    data = sortData(importData())
    '''
    max_diff = {}
    prev = {}
    names = {"Agata Sitko", "Austin Perkins #1", "Joseph Borenstein", "Alba Boström", "Bobb Matthews"}  # Add desired names here
    for row in data:
        name, value, age  = row[0], row[3], row[5]  
        if name in names:
            if name in prev:
                diff = float(value) - prev[name]
                max_diff[name] = max(max_diff.get(name, float('-inf')), diff)
            prev[name] = float(value)
        else:
            prev.pop(name, None)  # Reset previous value if encountering another lifter
  
                    
    print(max_diff)  # Output max differences for all specified lifters
    '''
    #data = filterNamesWithAtLeastTwoOccurrences(data) #again, this is removed due to updateDates. For more info, see the function.
    
    filtered = updateDates(data)
    transform_and_save_arrays(filtered)
    print("Data has been preprocessed and saved to files.")
    
    