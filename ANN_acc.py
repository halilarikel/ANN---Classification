

from keras.models import Sequential
import pandas as pd 
from keras.layers import Dense, Dropout
import numpy as np


#read in training data
data = pd.read_csv('diabetes_data.csv')

data_len =len(data['diabetes'])
x_train = data.loc[:data_len*0.8,:"age"]
y_train = data.loc[:data_len*0.8,"diabetes"]

x_test = data.loc[data_len*0.8+1:, :"age"]
y_test = data.loc[data_len*0.8+1:,"diabetes"]


#validation_split=0.2,

# Evaluating the ANN

def build_classifier():
    classifier = Sequential() # initialize neural network
    classifier.add(Dense(units = 100, kernel_initializer = 'uniform', activation = 'sigmoid', input_dim = x_train.shape[1]))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(units = 100, kernel_initializer = 'uniform', activation = 'sigmoid', input_dim = x_train.shape[1]))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(units = 100, kernel_initializer = 'uniform', activation = 'sigmoid', input_dim = x_train.shape[1]))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer ="adam", loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier



classifier = build_classifier()


classifier.fit(x_train, y_train, epochs=200, batch_size=48)


y_predict = classifier.predict(x_test)

for i in range(len(y_predict)):
    if y_predict[i] >= 0.6:
        y_predict[i] = 1
    else:
        y_predict[i] = 0

correct = 0
y_q_pre = np.array(y_predict)  
y_test_arr = np.array(y_test)
for i in range(len(y_q_pre)):
    if y_q_pre[i] == y_test_arr[i]:
        correct +=1
        
print('Acc:',(correct/len(y_predict))*100)