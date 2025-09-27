# We’ll use the classic Heart Disease dataset (UCI Cleveland).
# It has columns like:
# age, sex, cp (chest pain type),
# trestbps (resting blood pressure),
# chol (cholesterol),
# thalach (max heart rate),
# target → 1 = disease, 0 = no disease.

# Goal:

# Load and clean a healthcare dataset

# Train Logistic Regression, Decision Tree, k-NN

# Evaluate with accuracy, precision, recall

# Compare results

# Train all 3 models and compare metrics.
# Try different hyperparameters:
# max_depth for DecisionTree
# n_neighbors for k-NN
# penalty or C for Logistic Regression
# Which model gives the best tradeoff between accuracy, precision, recall?
# Bonus: Plot a bar chart of accuracy scores for all models.


# Setup Imports
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix

import kagglehub
import pandas as pd
import os

# Download dataset (latest version)
dataset_dir = kagglehub.dataset_download("ritwikb3/heart-disease-cleveland")

# Assume there is only one CSV in the folder
csv_files = [f for f in os.listdir(dataset_dir) if f.endswith(".csv")]
csv_path = os.path.join(dataset_dir, csv_files[0])

# Load Dataset
# Load CSV into DataFrame
df = pd.read_csv(csv_path)

# Quick check
print(df.head())
print(df.isna().sum())
# print("CSV file full path:", csv_path)

# Features & Target
X = df.drop(columns="target", axis=1)
Y = df["target"]

# Train-Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
# stratify=y preserves the class distribution across train/test splits

# Train Models

# Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, Y_train)
y_pred_log = log_reg.predict(X_test)
# Workflow of logistic regression:
# Start with random w and b
# Compute predicted probability p = sigmoid(w*x + b)
# Compute loss (cross-entropy)
# Compute gradient (slope) w.r.t. w and b
# Update w and b (step downhill)
# Repeat many times (max_iter) → eventually w and b converge

# Decision Tree
tree = DecisionTreeClassifier(max_depth=5, random_state=42)
tree.fit(X_train, Y_train)
y_pred_tree = tree.predict(X_test)

# k-NN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, Y_train)
y_pred_knn = knn.predict(X_test)

# Experiment with k(n_negihbors)
for k in [1, 3, 5, 6, 8, 10, 12, 14, 16]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, Y_train)
    print(f"k={k} → Accuracy:", knn.score(X_test, Y_test))

# Evaluate Models
def evaluate_model(y_true, y_pred, name):
  print(f"--- {name} ---")
  print("Accuracy Score", accuracy_score(y_true, y_pred))
  print("Precision Score", precision_score(y_true, y_pred))
  print("Recall Score", recall_score(y_true, y_pred))
  print("Confusion Matrix:", confusion_matrix(y_true, y_pred))

evaluate_model(Y_test, y_pred_log, "Logistic Regression")  
evaluate_model(Y_test, y_pred_tree, "Decision Tree")
evaluate_model(Y_test, y_pred_knn, "k-NN")

results = {
  "Models": ["Logistic Regression", "Decision Tree", "K-NN"],
  "Accuracy": [accuracy_score(Y_test, y_pred_log), accuracy_score(Y_test, y_pred_tree), accuracy_score(Y_test, y_pred_knn)],
  "Precision": [precision_score(Y_test, y_pred_log), precision_score(Y_test, y_pred_tree), precision_score(Y_test, y_pred_knn)],
  "Recall": [recall_score(Y_test, y_pred_log), recall_score(Y_test, y_pred_tree), recall_score(Y_test, y_pred_knn)]
}

results_df = pd.DataFrame(results)
print(results_df)

#                 Models  Accuracy  Precision    Recall
# 0  Logistic Regression  0.868852   0.812500  0.928571
# 1        Decision Tree  0.786885   0.727273  0.857143
# 2                 K-NN  0.737705   0.800000  0.571429


# Models and their metrics
models = ["Logistic Regression", "Decision Tree", "k-NN"]
accuracy = [0.868852, 0.786885, 0.737705]
precision = [0.8125, 0.727273, 0.8]
recall = [0.928571, 0.857143, 0.571429]

# Simple grouped bar chart using pandas (optional for simplicity)
df = pd.DataFrame({
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall
}, index=models)

df.plot(kind='bar', figsize=(8,5), ylim=(0,1), color=['skyblue', 'lightgreen', 'salmon'])
plt.title("Model Comparison: Accuracy, Precision, Recall")
plt.ylabel("Score")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# Summary:

# Logistic Regression has the best overall tradeoff: high accuracy (0.869), precision (0.813), and recall (0.929).  
# Decision Tree is lower on all metrics.  
# k-NN has decent precision (0.800) but very low recall (0.571), missing many positives.  
# Use Logistic Regression for overall balance.  
# Use k-NN if high precision is more important.


# Guidelines: When to use which

# If you have labeled data (supervised):
# Linear boundary: Logistic Regression
# Nonlinear patterns: Decision Tree or k-NN
# Large dataset: Logistic Regression or Decision Tree (k-NN slows down with many samples)
# Need probability estimates: Logistic Regression
# If you have unlabeled data (unsupervised):
# Use K-Means to find groups or patterns.
# Performance trade-offs:
# Logistic Regression → fast, interpretable, good baseline
# Decision Tree → can overfit, visualize, handles complex data
# k-NN → simple but slow on large data, sensitive to features
# K-Means → exploratory analysis, no labels required