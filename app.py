import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# -----------------------------
# App Title
# -----------------------------
st.title("🏠 House Price Prediction App")

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("kc_house_data.csv")

df = load_data()

# -----------------------------
# Data Preparation (SAME LOGIC)
# -----------------------------
X = df[["bedrooms", "waterfront", "floors"]]
y = df["price"]

data = pd.concat([X, y], axis=1).dropna()
X = data[["bedrooms", "waterfront", "floors"]]
y = data["price"]

# -----------------------------
# Train Linear Regression Model
# -----------------------------
model = LinearRegression()
model.fit(X, y)

st.success("Model trained successfully")

# -----------------------------
# User Inputs
# -----------------------------
bedrooms = st.number_input("Number of Bedrooms", min_value=0, step=1)
waterfront = st.selectbox("Waterfront (0 = No, 1 = Yes)", [0, 1])
floors = st.number_input("Number of Floors", min_value=0.0, step=0.5)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict House Price"):
    input_data = np.array([[bedrooms, waterfront, floors]])
    prediction = model.predict(input_data)
    st.success(f"Predicted House Price: ₹ {prediction[0]:,.2f}")
