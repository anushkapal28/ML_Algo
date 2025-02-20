import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


df = pd.read_csv("calories.csv")
print(df.head())

X = df[['User_ID']]
y = df['Calories']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared Score (R2): {r2}")

plt.scatter(X_test, y_test, color='red', label="Actual")
plt.plot(X_test, y_pred, color='blue', label="Predicted")
plt.xlabel("Time Spent (minutes)")
plt.ylabel("Calories Burned")
plt.title("Simple Linear Regression - Calories Burned vs Time Spent")
plt.legend()
plt.show()
