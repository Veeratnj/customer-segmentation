import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the scaler and model from disk
scaler_loaded = joblib.load('scaler.pkl')
kmeans_loaded = joblib.load('kmeans_model.pkl')

# Streamlit interface
st.title('Customer Segmentation App')
st.write('This app predicts customer segments based on Recency, Frequency, and Monetary values.')

# User inputs for Recency, Frequency, and Monetary
recency = st.number_input('Enter Recency (in days):', min_value=0)
frequency = st.number_input('Enter Frequency (number of transactions):', min_value=0)
monetary = st.number_input('Enter Monetary (total spend):', min_value=0)

# Button to predict
if st.button('Predict Segment'):
    # Create DataFrame from user input
    new_data = pd.DataFrame({
        'Recency': [recency],
        'Frequency': [frequency],
        'Monetary': [monetary]
    })
    
    # Scale the new data using the loaded scaler
    new_data_scaled = scaler_loaded.transform(new_data)
    
    # Predict the cluster for the new data using the loaded KMeans model
    prediction = kmeans_loaded.predict(new_data_scaled)
    
    # Show prediction
    st.write("Predicted Cluster:", prediction[0])

    # Show message based on cluster
    if prediction[0] == 0:
        st.write("This customer belongs to Cluster 0, which represents low-spending and frequent customers.")
    elif prediction[0] == 1:
        st.write("This customer belongs to Cluster 1, which represents high-spending and infrequent customers.")
    elif prediction[0] == 2:
        st.write("This customer belongs to Cluster 2, which represents moderate-spending and occasional customers.")
    elif prediction[0] == 3:
        st.write("This customer belongs to Cluster 3, which represents new or one-time customers.")
