# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 22:49:45 2024

@author: DELL
"""

import numpy as np
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler


loaded_model = pickle.load(open('C:/Users/DELL/Downloads/ml_deploy/trained_model.sav','rb'))
 
#creating a function
def weekly_sales_prediction(input_data):
    
    input_data_as_np_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_np_array.reshape(1,-1)
    #std_data = scaler.transform(input_data_reshaped)
    new_scaler = StandardScaler()
    new_scaler.fit(input_data_reshaped)
    std_data = new_scaler.transform(input_data_reshaped)
    print(std_data)
    prediction = loaded_model.predict(std_data)
    print(prediction)
    return prediction
 
def main():
    st.title('Walmart Sales Prediction')
    Store=st.text_input('Enter Store Number')
    Holiday_flag=st.text_input('Enter Holiday Flag')
    Temperature=st.text_input('Enter Temperature')
    Fuel_Price=st.text_input('Enter Fuel Price')
    CPI=st.text_input('Enter CPI')
    Unemployment=st.text_input('Enter Unemployment')
    
    Weekly_sales=''
    if st.button("Weekly Sales"):
        Weekly_sales=weekly_sales_prediction([Store,Holiday_flag,Temperature,Fuel_Price,CPI,Unemployment])
        
    st.success(Weekly_sales)


if __name__=='__main__':
    main()



