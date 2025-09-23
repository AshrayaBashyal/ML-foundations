# Classification with Logistic Regression
# Regression → predicts continuous values (house price, exam score).
# Classification → predicts discrete labels (spam / not spam, sick / healthy, pass / fail).

# Logistic Regression Idea
# It predicts probabilities using the sigmoid function:

# 𝜎(𝑧)=1/(1+e)^(-z)

# Output is between 0 and 1 (probability).
# If probability > 0.5 → class 1
# Else → class 0


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


