import streamlit as st 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.title("Placement Package Predictor")
try:
    df = pd.read_csv("placement.csv")
    

    X = df['cgpa'].values.reshape(-1, 1)
    y = df['package'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    

    def predict_package(cgpa):
        cgpa = np.array(cgpa).reshape(-1, 1)
        predicted_package = model.predict(cgpa)
        return predicted_package[0]

    cgpa_input = st.text_input("Enter CGPA:")
    if cgpa_input:
        try:
            cgpa_value = float(cgpa_input)
            predicted_package = predict_package(cgpa_value)
            st.write(f"### Predicted Placement Package: **{predicted_package:.2f} LPA**")
        except ValueError:
            st.error("Please enter a valid numeric CGPA.")

except FileNotFoundError:
    st.error("⚠️ Error: `placement.csv` file not found! Please make sure the file is in the same directory.")




