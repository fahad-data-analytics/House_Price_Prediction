import streamlit as st
import pickle
import numpy as np

# Load model + scaler
data = pickle.load(open("model.pkl", "rb"))
model = data["model"]
scaler = data["scaler"]

st.title("🏠 House Price Prediction App")

# Inputs
income = st.number_input("Average Income (in USD)", min_value=0.0)
age = st.number_input("House Age (Years)", min_value=0.0)
rooms = st.number_input("Number of Rooms", min_value=0.0)
population = st.number_input("Population in Area", min_value=0.0)

if st.button("Predict Price"):
    features = np.array([[income, age, rooms, population]])
    scaled_features = scaler.transform(features)  # ✅ Apply scaling
    prediction = model.predict(scaled_features)[0]
    st.success(f"Predicted House Price: ${prediction:,.2f}")

