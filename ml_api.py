 # -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle


loaded_model= pickle.load(open('C:/Intel/Deploying ml model/machine learning/diabetes_model.sav','rb'))

input_data = [10,168,74,0,0,38,0.537,34]
input_data_as_numpy_array = np.asarray(input_data)

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]==0):
    print('person is non diabetic')
else:
    print ('person is diabetic')
