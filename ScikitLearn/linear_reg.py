# ---------------------- Step 0: Imports ----------------------
import numpy as np
import pandas as pd
from sklearn.model_selection import  train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ---------------------- Step 1: Data -------------------------
data = {
    "age": [25, 30, 35, 32, 40, 50, 55, 60, np.nan],
    "cholesterol": [200, 210, 215, 220, 230, 250, 260, 270, 240],
    "sex": ["male", "female", "female", "male", "female", "male", "male", "female", "male"],
    "disease": [0, 0, 0, 1, 0, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

print("\n\nOriginal Data:\n", df)
print(f"\n\nMissing Values:\n{df.isna().sum()}")

df["age"]=df["age"].fillna(df["age"].mean())

print("\n\nCleaned Data:\n", df)

X = df[["age"]]
Y = df["cholesterol"]

# ---------------------- Step 2: Split Data -------------------
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle=True)

# ---------------------- Step 3: Train Model -----------------
model = LinearRegression()
model.fit(X_train, Y_train)

# ---------------------- Step 4: Predict ----------------------
Y_pred = model.predict(X_test)

df_pred = X_test.copy()
df_pred["predicted_cholesterol"] = Y_pred
df_pred["Actual_cholesterol"] = Y_test.values
print("\nPredicted cholesterol and actual cholesterol for test set:\n", df_pred)

# ---------------------- Step 5: Evaluate ---------------------
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

print("\n\nMean Square Error:\n", mse)
print("\n\nCoefficient of Determination, r^2:\n", r2)

# ---------------------- Step 6: Optional User Prediction -----
while True:
    user_input = input("\nEnter the age for which the cholesterol is to be predicted (or 'q' to quit):\n")
    if user_input.lower() == 'q':
        break
    try:
        age_input = float(user_input)
        y_pred = model.predict([[age_input]])
        print(f"The predicted cholesterol for age {age_input} = {y_pred[0]:.2f}\n")
    except ValueError:
        print("Please enter a valid number or 'q' to quit.")

# ---------------------- Step 7: Visualize ---------------------
plt.scatter(X, Y, color="blue", label="Data")
plt.plot(X, model.predict(X), color="red", label="Regression Line")
plt.xlabel("Age")
plt.ylabel("Cholesterol")
plt.title("Linear Regression: Age vs Cholesterol")
plt.legend()
plt.show()