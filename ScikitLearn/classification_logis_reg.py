# Classification with Logistic Regression
# Regression → predicts continuous values (house price, exam score).
# Classification → predicts discrete labels (spam / not spam, sick / healthy, pass / fail).

# Logistic Regression Idea
# It predicts probabilities using the sigmoid function:

# 𝜎(𝑧)=1/(1+e)^(-z)

# Output is between 0 and 1 (probability).
# If probability > 0.5 → class 1
# Else → class 0

# 1. Accuracy Score

# Definition:
# Accuracy = fraction of correct predictions.

# Accuracy=Correct Predictions/Total Predictions
# Example:
# Suppose we predict whether patients have a disease (0 = No, 1 = Yes).
# Predictions: [0, 1, 1, 0]
# Actual: [0, 0, 1, 0]
# Correct = 3 out of 4 → Accuracy = 0.75 (75%).


# 2. Confusion Matrix

# Definition:
# A table that shows how many predictions fell into each category.

# For binary classification (0/1):
# 	                 Predicted No (0)	      Predicted Yes (1)
# Actual No (0)	    True Negative (TN)	     False Positive (FP)
# Actual Yes (1)	False Negative (FN)	     True Positive (TP)


# 3. Classification Report

# This gives precision, recall, F1-score for each class.

# Precision: Out of all predicted positives, how many were correct?
# Precision=𝑇𝑃/(𝑇𝑃+𝐹𝑃) 	​

# Recall (Sensitivity): Out of all actual positives, how many did we catch?
# Recall=𝑇𝑃/(TP+FN)​

# F1 Score: Harmonic mean of Precision & Recall (balances them).
# 𝐹1=2 x Precision×Recall /(Precision + Recall)


# Precision → How reliable are positive predictions?
# Recall → How many actual positives did we catch?
# F1 → Balance between the two.
# Support → Number of actual samples in that class.

# ---------------------- Step 0: Imports ----------------------
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report 

# ---------------------- Step 1: Data -------------------------
data = {
    "age": [25, 30, 35, 40, 50, 55, 60, 65],
    "cholesterol": [200, 210, 220, 230, 250, 260, 270, 280],
    "disease": [0, 0, 0, 1, 1, 1, 1, 1]  # 0 = No, 1 = Yes
}
df = pd.DataFrame(data)

# ---------------------- Step 2: Split ------------------------
X = df[["age", "cholesterol"]]
Y = df["disease"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=42) 

# ---------------------- Step 3: Train Model -----------------
model = LogisticRegression()
model.fit(X_train, Y_train)

# ---------------------- Step 4: Predict ----------------------
Y_pred = model.predict(X_test)

# ---------------------- Step 5: Evaluate ---------------------
acc = accuracy_score(Y_test, Y_pred)
cm = confusion_matrix(Y_test, Y_pred)
report = classification_report(Y_test, Y_pred)

print("Predictions:", Y_pred)
print("Accuracy:", acc)
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", report)



# Mini-Tasks:
# Load your heart.csv dataset.
# Train a Logistic Regression model to predict disease (target).
# Use features: age, cholesterol.
# Print accuracy, confusion matrix, and classification report.
# Try adding sex (convert male/female → 0/1) and retrain.
# Bonus: Compare accuracy with only age vs with age + cholesterol.


data = pd.read_csv("C:\\Users\\Ashraya Bashyal\\Desktop\\ML\\ScikitLearn\\heart.csv")
df = pd.DataFrame(data)

print(df.head())
print(df.isna().sum())

X = df[["age", "cholesterol"]]
Y = df["target"]    #target --> (disease)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, random_state=42)

model = LogisticRegression()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)


df_pred = X_test.copy()
df_pred["predicted_disease"] = Y_pred
df_pred["Actual_disease"] = Y_test.values
print("\nPredicted disease and actual disease for test set:\n", df_pred)

acc = accuracy_score(Y_test, Y_pred)
cm = confusion_matrix(Y_test, Y_pred)
report = classification_report(Y_test, Y_pred)

print("\n\nAccuracy Score:\n",acc)
print("\n\nConfusion Matrix:\n",cm)
print("\n\nClassification Report:\n",report)


print("\n\n\nModel with features: age, cholesterol and sex:\n\n")

df["sex"] = df["sex"].map({"Female": 0, "Male": 1})

X = df[["age", "cholesterol", "sex"]]
Y = df["target"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

df_pred = X_test.copy()
df_pred["predicted_disease"] = Y_pred
df_pred["Actual_disease"] = Y_test.values
print("\n\nPredicted disease and actual disease for test set:\n", df_pred)


X1 = df[["age", "cholesterol"]]
X2 = df[["age"]]
Y = df["target"]


X1_train, X1_test, Y_train, Y_test = train_test_split(X1, Y, test_size=0.2, random_state=42)
X2_train, X2_test, Y_train, Y_test = train_test_split(X2, Y, test_size=0.2, random_state=42)

model1 = LogisticRegression()
model2 = LogisticRegression()
model1.fit(X1_train, Y_train)
model2.fit(X2_train, Y_train)
Y1_pred = model.predict(X_test)
Y2_pred = model.predict(X_test)

acc1 = accuracy_score(Y_test, Y1_pred)
acc2 = accuracy_score(Y_test, Y2_pred)

print(f"\n\nAccuracy with only age:{acc2} and with age + cholesterol:{acc1}")

