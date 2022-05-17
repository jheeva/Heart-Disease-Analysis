# -*- coding: utf-8 -*-
"""
Created on Wed May 18 03:08:06 2022

@author: End User
"""

#%%packages
#from tensorflow.keras.models import load_model
import pickle
import os
import numpy as np
import streamlit as st
import pandas as pd


# static values
MMS_SCALER_SAVE_PATH = os.path.join(os.getcwd(), 'saved_models', 'mms_scaler.pkl')
MODEL_SAVE_PATH=os.path.join(os.getcwd(), 'saved_models', 'model_scaler.pkl')



# MinMaxScaler
with open(MMS_SCALER_SAVE_PATH, "rb") as f:
    minmax = pickle.load(f)
    
#RandomForestClassifier
with open(MODEL_SAVE_PATH, "rb") as r:
    forest = pickle.load(r)
    
heart_attack_disease = {0: "chances of getting heart attack is low",
                 1: "chances of getting heart attack is more"}

#%% Deployment

patients_info = pd.DataFrame({"age": [65,61,45,40,48,41,36,45,57,69],
                 "sex": [1,1,0,0,1,1,0,1,1,1],
                 "cp": [3,0,1,1,2,0,2,0,0,2],
                 "trtbps": [142,140,128,125,132,108,121,111,155,179],
                 "chol": [220,207,204,307,254,165,214,198,271,273],
                 "fbs": [1,0,0,0,0,0,0,0,0,1],
                 "restecg": [0,0,0,1,1,0,1,0,0,0],
                 "thalachh": [158,138,172,162,180,115,168,176,112,151],
                 "exng": [0,1,0,0,0,1,0,0,1,1],
                 "oldpeak": [2.3,1.9,1.4,0,0,2,0,0,0.8,1.6],
                 "slp": [1,2,2,2,2,1,2,2,2,1],
                 "caa": [0,1,0,0,0,0,0,1,0,0],
                 "thall": [1,3,2,2,2,3,2,2,3,3],
                 "True output": [1,0,1,1,1,0,1,0,0,0]
                 })





# splitting
X_true = patients_info.drop(labels=["True output"], axis=1)
y_true = np.expand_dims(patients_info["True output"], -1)

# evaluate model with new data
patient_accuracy = forest.score(X_true, y_true)
print(f"Accuracy for Model Prediction with new data is {patient_accuracy:.0%}")
#Predict the data one by one


new_pred = forest.predict(X_true)

#loop
if np.argmax(new_pred) == 0:
    new_pred = [0,1]
    print(heart_attack_disease[np.argmax(new_pred)])
else:
    pred_new = [1,0]
    print(heart_attack_disease[np.argmax(new_pred)])

#%% Streamlit 
with st.form('Heart Disease Prediction Form'):
    st.write("Patient's Info")
    age = int(st.number_input("Age")) # add int because not float
    sex = int(st.number_input("Sex")) # not need int because of float value
    cp = int(st.number_input("Chest Pain type"))
    trtbps = int(st.number_input("Resting blood pressure (in mm Hg)"))
    chol = int(st.number_input("Cholestoral in mg/dl fetched via BMI sensor"))
    fbs = int(st.number_input("(Fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)"))
    restecg = int(st.number_input("Resting electrocardiographic"))
    thalach = int(st.number_input("Maximum heart rate achieved"))
    exng = int(st.number_input("Exercise induced angina (1 = yes; 0 = no)"))
    oldpeak = st.number_input("Previous peak")
    slp = int(st.number_input("Slp"))
    ca = int(st.number_input("Number of major vessels (0-3)"))
    thall = int(st.number_input("Thall"))
    
    submitted = st.form_submit_button('Submit')
    
    if submitted == True:
        patience_columns = np.array([age,sex,cp,trtbps,chol,fbs,restecg,thalach,
                                  exng,oldpeak,slp,ca,thall])
                                  
        patience_columns = minmax.transform(np.expand_dims(patience_columns, axis=0))
        new_pred_2 = forest.predict(patience_columns)
        if np.argmax(new_pred_2) == 1:
            st.success(f"You have {heart_attack_disease[np.argmax(new_pred_2)]}")
        else:
            st.warning(f"You have {heart_attack_disease[np.argmax(new_pred)]}")
    

   