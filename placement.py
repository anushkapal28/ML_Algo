import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.title("Placement Package Predictor")

uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.write(df.head())

    X = df['cgpa'].values.reshape(-1, 1)
    y = df['package'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.write(f"**Mean Squared Error:** {mse:.2f}")
    st.write(f"**R-squared Value:** {r2:.2f}")

    fig, ax = plt.subplots()
    ax.scatter(X_test, y_test, color='blue', label="Actual Data")
    ax.plot(X_test, y_pred, color='red', label="Predicted Line")
    ax.set_title('CGPA vs Placement Package')
    ax.set_xlabel('CGPA')
    ax.set_ylabel('Placement Package')
    ax.legend()
    st.pyplot(fig)

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
else:
    st.write("Please upload a CSV file to proceed.")
