# Classification with k-NN
#-----------------------------------------
# What is k-NN?

# k-NN = k-Nearest Neighbors
# It doesn’t build equations or trees. Instead:
# Store all training points.
# When predicting, find the k closest neighbors (using distance, usually Euclidean).
# Take a majority vote among neighbors → prediction.

# Example:
# If k=3 and among 3 nearest patients, 2 have disease, 1 doesn’t → predict disease.

# Small k → more flexible, but risk of noise/overfitting.
# Large k → smoother boundaries, but may miss details.


# Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Dataset
data = {
    "age": [25, 30, 35, 40, 50, 55, 60, 65],
    "cholesterol": [200, 210, 220, 230, 250, 260, 270, 280],
    "disease": [0, 0, 0, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

X = df[["age", "cholesterol"]]
y = df["disease"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# rain k-NN Classifier
knn = KNeighborsClassifier(n_neighbors=3)  # k=3
knn.fit(X_train, y_train)

# Make Predictions
y_pred = knn.predict(X_test)
print("Predictions:", y_pred)

# Evaluate Model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Experiment with k
for k in [1, 3, 5]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    print(f"k={k} → Accuracy:", knn.score(X_test, y_test))

