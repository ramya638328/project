import streamlit as st
import numpy as np
import pickle

with open("house_prediction.pkl", "rb") as f:
    model = pickle.load(f)

st.title("House Price Prediction")

bedrooms = st.number_input("Bedrooms", 0, 10)
waterfront = st.selectbox("Waterfront", [0, 1])
floors = st.number_input("Floors", 0.0, 5.0)

if st.button("Predict"):
    pred = model.predict([[bedrooms, waterfront, floors]])
    st.success(f"Predicted Price: ₹ {pred[0]:,.2f}")
