import streamlit as st
import pickle
import numpy as np

# Load trained model
with open("house_prediction.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🏠 House Price Prediction App")

st.write("Enter house details to predict price")

# User inputs
bedrooms = st.number_input("Number of Bedrooms", min_value=0, step=1)
waterfront = st.selectbox("Waterfront (0 = No, 1 = Yes)", [0, 1])
floors = st.number_input("Number of Floors", min_value=0.0, step=0.5)

# Prediction button
if st.button("Predict Price"):
    input_data = np.array([[bedrooms, waterfront, floors]])
    prediction = model.predict(input_data)
    st.success(f"Estimated House Price: ₹ {prediction[0]:,.2f}")
