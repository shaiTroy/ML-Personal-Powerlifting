import csv
from datetime import datetime
import numpy as np
import sys
from IsoForest import Iso

def importData():
    with open('openipf-2025-03-15-e5a42b06.csv', mode='r') as file:
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

def exportData(data, name):
    # This function is not used anymore, keeping it to show past use
    with open(name, mode='w') as file:
        file.write(f"Name, Sex, Weight, Total, Date\n")
        for row in data:
            for i in range(5):
                file.write(row[i])
                if i < 4: file.write(', ')
            file.write('\n')
            
def filterNamesWithAtLeastTwoOccurrences(data):
    '''
    This function has become useless after implementation of updateDates.
    It was used to filter out lifters with only one competition, but updateDates is used to ensure lifters dont have two competitions on the same day.
    If a lifter has two competitions on the same day, we want to remove duplicates. If a lifter only has one competition,
    the algo will reckognize that the next lifter has the first lifting date to 0 aswell.
    Therefore, this function is not needed, but is kept for understanding why RNNs wont work.
    '''
    name_count = {}
    
    for row in data:
        name = row[0]
        if name in name_count:
            name_count[name] += 1
        else:
            name_count[name] = 1

    
    # for finding the number of lifters with atleast x competitions
    # used to understand why a RNN isnt a good choice for this data
    '''
    values = list(name_count.values())
    bins = range(min(values), max(values) + 2)
    holder = 0
    for i in range(len(bins) - 1, 0, -1):
        holder += len([name for name in name_count if name_count[name] == bins[i]])
        print(f"Found {holder} lifters with atleast {bins[i]} competitions.")
    '''

    filtered_data = [row for row in data if name_count[row[0]] >= 2]
    print(f"Filtered out {len(data) - len(filtered_data)} entries, {len(filtered_data)} entries remaining.")
    return filtered_data

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
    
    #filterNamesWithAtLeastTwoOccurrences(filtered) # This is can be uncommented to check if the filter function removes 0 names

    #filter out entries with no age or incorrect attempts
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
            holder = [1, row[1], float(row[2]), float(row[3]), float(row[4]), row[5], float(row[8]), float(row[9]), float(row[10]), float(row[11]), float(row[12]), float(row[13]), float(row[14]), float(row[15]), float(row[16]), float(row[17]), float(row[18]), float(row[19]), row[20]]
        else:
            if holder[1] == 'M':
                holder[1] = 1
            else:
                holder[1] = 0
            holder[4] = row[4] - holder[4]
            holder.append(float(row[2])-holder[2])
            holder.append(float(row[3])-holder[3])
            comps.append(holder)
            holder = [1, row[1], float(row[2]), float(row[3]), float(row[4]), row[5], float(row[8]), float(row[9]), float(row[10]), float(row[11]), float(row[12]), float(row[13]), float(row[14]), float(row[15]), float(row[16]), float(row[17]), float(row[18]), float(row[19]), row[20]]
    
    print(f"Number of entries: {len(comps)}")

    #Send to unsupervised algo to learn injured vs healthy
    Inputs, Outputs = Iso(comps)
    
    #Array looks as following:
    
    #Inputs
    #[0] = 1 (bias)
    #[1] = Sex (0 for F, 1 for M)
    #[2] = Init Weight
    #[3] = Init Total
    #[4] = Days between competitions
    #[5] = Age
    #[6] = Squat1
    #[7] = Squat2
    #[8] = Squat3
    #[9] = Best Squat
    #[10] = Bench1
    #[11] = Bench2
    #[12] = Bench3
    #[13] = Best Bench
    #[14] = Deadlift1
    #[15] = Deadlift2
    #[16] = Deadlift3
    #[17] = Best Deadlift
    #[18] = GL
    #[19] = Change Weight

    
    #Outputs
    #[0] = Change Total
    
    
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
    
    