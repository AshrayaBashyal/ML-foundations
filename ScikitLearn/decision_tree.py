# What is a Decision Tree?

# Think of a flowchart of yes/no questions.
# Example for disease prediction:
# Is cholesterol > 240?
# Yes → Predict Disease
# No → Check Age
# Age > 50 → Predict Disease
# Otherwise → No Disease
# Decision Trees split data into branches until they reach a decision.


# How it Works

# The algorithm looks at your data.
# It chooses the best feature to split (using math like Gini index, Entropy, or MSE).
# Example: “Split by age > 45” if that separates healthy vs diseased well.
# It keeps splitting data into smaller groups (recursively).
# Stops when:
# All samples in a group are pure (same class), or
# Tree reaches max depth / min samples per leaf (to prevent overfitting).


# Linear Regression, Logistic Regression → linear boundary (straight line).
# Decision Tree → non-linear boundaries (piecewise, flexible).
# Trees can overfit if grown too deep.
# Overfitting means:
# Your model learns the training data too well (including noise, outliers, random quirks).
# As a result, it does great on training data but fails on new data (bad generalization).


# When to Use Each

# Linear Regression → Continuous outcomes, linear relationship (e.g., house price vs area).
# Logistic Regression → Binary classification with linear separation (e.g., pass/fail if marks > 40).
# Decision Tree → Complex, non-linear decision rules (e.g., medical diagnosis, loan approval).



# Import Libraries
import pandas as pd
from sklearn.model_selection import  train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Dataset
data = pd.read_csv(r"C:\Users\Ashraya Bashyal\Desktop\ML\ScikitLearn\heart.csv")

df = pd.DataFrame(data)

X = df[["age", "cholesterol"]]
Y = df["target"]   # target <--> disease

# Train-Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, random_state=42)

# Train Decision Tree
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X_train, Y_train)

# Predictions
Y_pred = tree.predict(X_test)

# Evaluate Model
acc = accuracy_score(Y_test, Y_pred)
cm = confusion_matrix(Y_test, Y_pred)
report = classification_report(Y_test, Y_pred)

print("Predictions:", Y_pred)
print("Accuracy:", acc)
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", report)

# Visualize the Tree
plt.figure()
plot_tree(tree, feature_names=["age", "cholesterol"], class_names=["No Disease", "Disease"], filled=True)
plt.show()


# Gini=1 − (i=0 to i=K ∑(Pi)^2)

# Example: value = [15, 5]

# Total samples
# total = 15 + 5
# total = 20

# Proportions
# p1 = 15 / 20   # = 0.75 (No)
# p2 = 5 / 20    # = 0.25 (Yes)

# Gini calculation
# gini = 1 - (p1**2 + p2**2)
# gini = 1 - (0.75**2 + 0.25**2)
# gini = 1 - (0.5625 + 0.0625)
# gini = 1 - 0.625
# gini = 0.375

# print(gini)  # 0.375
