# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 12:18:55 2025

@author: abasu
"""
import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the trained model
with open("ship_type_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load the scaler
with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Load the label encoders
with open("ship_type_label_encoder.pkl", "rb") as label_encoder_file:
    ship_type_label_encoder = pickle.load(label_encoder_file)

with open("label_encoders (1).pkl", "rb") as encoders_file:
    label_encoders = pickle.load(encoders_file)

# Define the list of input features (same as in training)
features = ["SOG", "Acceleration", "COG", "Width_Length_Ratio", 
            "Draught_Length_Ratio", "Navigational_status", "Destination"]

# Streamlit UI
st.title("🚢 Ship Type Prediction App")
st.write("### Enter the ship details below to predict the Ship Type.")

# Create user input fields
sog = st.number_input("Speed Over Ground (SOG)", min_value=0.0, step=0.1)
acceleration = st.number_input("Acceleration", min_value=-10.0, step=0.1)
cog = st.number_input("Course Over Ground (COG)", min_value=0.0, max_value=360.0, step=0.1)
width_length_ratio = st.number_input("Width to Length Ratio", min_value=0.0, step=0.01)
draught_length_ratio = st.number_input("Draught to Length Ratio", min_value=0.0, step=0.01)
navigational_status = st.selectbox("Navigational Status", label_encoders["Navigational status"].classes_)
destination = st.selectbox("Destination", label_encoders["Destination"].classes_)

# When user clicks the "Predict Ship Type" button
if st.button("Predict Ship Type"):
    # Encode categorical inputs
    destination_encoded = label_encoders["Destination"].transform([destination])[0]
    nav_status_encoded = label_encoders["Navigational status"].transform([navigational_status])[0]

    # Prepare input data
    input_data = np.array([[sog, acceleration, cog, width_length_ratio, 
                            draught_length_ratio, nav_status_encoded, destination_encoded]])

    # Scale the numerical features
    input_data[:, :5] = scaler.transform(input_data[:, :5])  # Scale first 5 numeric features

    # Predict Ship Type
    ship_type_encoded = model.predict(input_data)[0]

    # Convert back to original label
    predicted_ship_type = ship_type_label_encoder.inverse_transform([ship_type_encoded])[0]

    # Display the result
    st.success(f"🚢 Predicted Ship Type: **{predicted_ship_type}**")

