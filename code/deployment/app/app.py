import streamlit as st
import requests

st.title("Model Prediction")

features_input = st.text_input("Enter features separated by commas:")

if st.button("Predict"):
    if features_input:
        features = [float(f) for f in features_input.split(",")]
        response = requests.post('http://model-api:6000/predict', json={'features': features})
        
        if response.status_code == 200:
            prediction = response.json()['prediction']
            st.success(f"Prediction: {prediction}")
        else:
            st.error("Error in prediction request")
    else:
        st.warning("Please enter features to predict.")
