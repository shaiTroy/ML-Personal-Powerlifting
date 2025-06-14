import numpy as np
'''
catagories:
    gender
    init weight
    Days between competitions 
    GL score
    change in weight

all entries are stored. 
After a prediction, we compare their input data to the input data of people in similar groups.
'''

def seperate(input_data, difference = 5, accuracy = 0):
    '''
    Calls the combined_data.npy file and compares the input data to the data in the file.
    for the first column, the input_data and combined_data.npy file should be the same.
    For every other column, if the difference between the input data and the data in the file is less than the difference, that data is used.
    Finally, we create bins of accuracy for the final three columns, which are the squat, bench and deadlift.
    The bins are between -15 and 35, in range of 5.
    '''
    
    
def save_data(data, accuracy_squat, accuracy_bench, accuracy_deadlift):
    '''
    Combines the data with the accuracy of the model and saves it to a file.
    Should only keep the following columns from the data:
    0, 1, 3, 17, 18
    '''
    data = data[:, [0, 1, 3, 17, 18]]
    data = np.column_stack((data, accuracy_squat, accuracy_bench, accuracy_deadlift))
    np.save('data/combined_data.npy', data)