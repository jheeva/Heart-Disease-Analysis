# -*- coding: utf-8 -*-
"""
Created on Wed May 18 02:03:55 2022

@author: End User
"""
import os
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout,Dense,BatchNormalization
#from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


DATA_PATH = os.path.join(os.getcwd(),'heart.csv')
MMS_SCALER_SAVE_PATH = os.path.join(os.getcwd(), 'saved_models', 'mms_scaler.pkl')
MODEL_SAVE_PATH=os.path.join(os.getcwd(), 'saved_models', 'model_scaler.pkl')



#%%EDA


#step 1)load data

'''
Read the file from it located(directory)
'''
df = pd.read_csv(DATA_PATH)


#step 2)intepret data
'''
get description of the data(eg:data type)
check for the NaN values
'''
df.info()
df.describe().T


#step3)data cleaning
# Drop duplicate
pure_data = df.drop_duplicates()
pure_data.info()
pure_data.duplicated().sum()
'''
if any duplicate data inside the dataset will be discard from the dataset
'''

# Check NaN value
msno.matrix(pure_data)
pure_data.isnull().sum() 
'''
to check dataset whether got any NaN values
'''



#%% Feature Selection
'''
# Using Lasso Regression
#correlation/Lasso
#X,y
#data intrepretation
'''

X = pure_data.drop(labels=["output"], axis=1)
y = np.expand_dims(pure_data["output"], axis=-1)


'''each column will be correlated'''
corr=df.corr()

''''visualize the correlation between columns '''
plt.figure(figsize=(12,10))
sns.heatmap(corr, annot=True, cmap=plt.cm.Blues)
plt.show()

#%% Feature Scaling

'''
plot the graph(MinMax Scaled)

'''
scaler=MinMaxScaler()
X_scaled=scaler.fit_transform(X)
sns.distplot(X_scaled)
plt.title("MinMax Scaled Data")
plt.legend()
plt.show()




pickle.dump(scaler,open('mms_scaler.pkl','wb'))

#%% Model Building


'''
sequential model
'''

model=Sequential()
model.add(Dense(64,activation=('relu'),input_shape=(14,)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(64,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(2,activation='softmax'))   

'''
softmax is that converting values to probability
'''
          
model.summary()

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics='acc')






#%% Model Configuration
from sklearn.ensemble import RandomForestClassifier

'''train_test_split'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=420)


'''  RandomForestClassifier  '''
forest = RandomForestClassifier().fit(X_train, y_train)

#%% Model Evaluation
patient_accuracy = forest.score(X_test, y_test)
print(f"Validation Accuracy for this estimator is {patient_accuracy:.00%}")


#%%model saving

model = pickle.dump(forest, open(MODEL_SAVE_PATH, "wb"))