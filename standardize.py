import numpy as np

class Standard:
    def __init__(self, Input_Train, Inputs_Test, Inputs_Validation=None):
        self.setNormalizer(Input_Train)
        if Inputs_Validation is None:
            self.normalize(Input_Train, Inputs_Test)
        else: self.normalize_w_valid(Input_Train, Inputs_Test, Inputs_Validation)
        
    def setNormalizer(self, x):
        self.mean = np.mean(x, axis=0, keepdims=True)
        self.std = np.std(x, axis=0, keepdims=True)
        self.std[self.std == 0] = 1

    def normalize_w_valid(self, Inputs_Train, Inputs_Test, Inputs_Validation):
        self.Inputs_Train = (Inputs_Train - self.mean) / self.std
        self.Inputs_Test = (Inputs_Test - self.mean) / self.std 
        self.Inputs_Validation = (Inputs_Validation - self.mean) / self.std
    
    def normalize(self, Inputs_Train, Inputs_Test):
        self.Inputs_Train = (Inputs_Train - self.mean) / self.std
        self.Inputs_Test = (Inputs_Test - self.mean) / self.std 
