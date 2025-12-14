import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# -----------------------------
# App Title
# -----------------------------
st.title("🏠 House Price Prediction App")

st.write("Upload the kc_house_data.csv file to train the model")

# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload kc_house_data.csv",
    type=["csv"]
)

if uploaded_file is not None:
    # Read dataset
    df = pd.read_csv(uploaded_file)

    st.success("Dataset loaded successfully")

    # -----------------------------
    # Data Preparation
    # -----------------------------
    X = df[["bedrooms", "waterfront", "floors"]]
    y = df["price"]

    data = pd.concat([X, y], axis=1).dropna()
    X = data[["bedrooms", "waterfront", "floors"]]
    y = data["price"]

    # -----------------------------
    # Train Model
    # -----------------------------
    model = LinearRegression()
    model.fit(X, y)

    st.success("Model trained successfully")

    # -----------------------------
    # User Inputs
    # -----------------------------
    st.subheader("Enter House Details")

    bedrooms = st.number_input("Bedrooms", min_value=0, step=1)
    waterfront = st.selectbox("Waterfront (0 = No, 1 = Yes)", [0, 1])
    floors = st.number_input("Floors", min_value=0.0, step=0.5)

    # -----------------------------
    # Prediction
    # -----------------------------
    if st.button("Predict House Price"):
        input_data = np.array([[bedrooms, waterfront, floors]])
        prediction = model.predict(input_data)
        st.success(f"Predicted House Price: ₹ {prediction[0]:,.2f}")

else:
    st.warning("Please upload kc_house_data.csv to continue")
