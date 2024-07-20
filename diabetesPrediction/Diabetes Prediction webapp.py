# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 16:56:12 2024

@author: HP
"""

import numpy as np
import pickle
import streamlit as st

# Loading the saved model
try:
    loaded_model = pickle.load(open('C:/Users/HP/Desktop/diabetesPrediction/trained_model.sav', 'rb'))
    st.write("Model loaded successfully.")
except Exception as e:
    st.write(f"Error loading model: {e}")

# Creating a function for prediction
def diabetes_prediction(input_data):
    try:
        # Changing the input_data to numpy array
        input_data_as_numpy_array = np.asarray(input_data, dtype=np.float64)  # Convert to float64

        # Reshape the array as we are predicting for one instance
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
        
        st.write(f"Input data reshaped: {input_data_reshaped}")

        # Make the prediction
        prediction = loaded_model.predict(input_data_reshaped)
        st.write(f"Prediction result: {prediction}")

        if prediction[0] == 0:
            return 'The person is not diabetic'
        else:
            return 'The person is diabetic'
    except Exception as e:
        st.write(f"Error during prediction: {e}")
        return 'Error during prediction'

def main():
    # Giving a title
    st.title('Diabetes Prediction Web App')
    
    # Getting the input data from the user
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure value')
    SkinThickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('Age of the Person')
    
    # Code for prediction
    diagnosis = ''
    
    # Creating a button for prediction
    if st.button('Diabetes Test Result'):
        try:
            # Convert inputs to float
            input_data = [
                float(Pregnancies),
                float(Glucose),
                float(BloodPressure),
                float(SkinThickness),
                float(Insulin),
                float(BMI),
                float(DiabetesPedigreeFunction),
                float(Age)
            ]
            
            st.write(f"Input data: {input_data}")
            
            diagnosis = diabetes_prediction(input_data)
        except ValueError as e:
            st.write(f"Input conversion error: {e}")
            diagnosis = 'Error in input conversion'
        
    st.success(diagnosis)
    
if __name__ == '__main__':
    main()

    
    
    