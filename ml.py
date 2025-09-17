# Linear Regression – Normal Equation
#--------------------------------------------------------------- 

# 1. Model Form (Single Feature)
#     y = w * x + b

#    Multiple Features:
#     y = w1*x1 + w2*x2 + ... + wn*xn + b

# 2. Matrix Form
#     y_hat = X @ theta

#     X = [
#         [1, x1(1), x2(1), ..., xn(1)],
#         [1, x1(2), x2(2), ..., xn(2)],
#         ...
#         [1, x1(m), x2(m), ..., xn(m)]
#     ]

#     theta = [
#         b,
#         w1,
#         w2,
#         ...,
#         wn
#     ]

#     For a single sample i:
#         y_hat(i) = b + w1*x1(i) + w2*x2(i) + ... + wn*xn(i)

# 3. Loss Function (Mean Squared Error)
#     J(theta) = (1/m) * sum_{i=1}^{m} (y(i) - y_hat(i))**2
#     y_hat(i) = x(i) @ theta

# 4. Normal Equation (Closed-Form Solution)
#     theta = (X.T @ X)^(-1) @ X.T @ y
#     Condition: X.T @ X must be invertible (full rank)
#     If singular: use Ridge Regression or Gradient Descent


# Data (Size in sqft vs Price in $1000):
#     Size = [1, 2, 3, 4]
#     Price = [1, 2, 2, 4]

# Goal:
#     Fit a linear regression model:
#         price = w * size + b

# Tasks:
# 1. Create the feature matrix X (with bias column of 1s)
# 2. Create the target vector y
# 3. Use Normal Equation to compute theta = [b, w]
# 4. Predict prices for given sizes: size = 5

import numpy as np

X = np.array([[1],[2],[3],[4]])
X_b = np.c_[np.ones((4, 1)), X]
print(f"X_b : {X_b}")
Y = np.array([[1],[2],[2],[4]])

theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ Y 
# theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(Y) 
print(f"theta : {theta}")

# price=1⋅size + 0 as m = 1 and b(intercept) = 0
size = 2 
X_new = np.array([[1, size]])
print(X_new)
Y_pred = X_new  @ theta
print(f"Predicted Price for size {size} => {Y_pred}")



# Linear Regression: Gradient Descent
#---------------------------------------------------------------------

# Gradient Descent (GD): optimization to minimize loss (MSE)
# Loss: J(θ) = (1/m) * Σ (y_hat - y)^2
# Update rule: θ = θ - α * ∇θ J(θ)
#   α = learning rate
#   ∇θ J(θ) = (2/m) * X.T @ (Xθ - y)   # gradient for linear regression


# Example: Same Dataset as Day 4
# Dataset:
# Size (x) | Price (y)
# 1        | 1
# 2        | 2
# 3        | 2
# 4        | 4
#
# Step 1: Setup Data
# - Define X (features) and y (targets)
# - Add bias term using np.c_
#
# Step 2: Initialize Parameters
# - Random init for theta (bias + slope)
# - Set learning_rate, n_iterations, m
#
# Step 3: Gradient Descent Loop
# - Compute gradients
# - Update theta
# - Print theta found
#
# Q: What theta values should you expect after GD?
#
# Step 4: Making Predictions
# - Predict price for new input [5]
# - Q: What output do you expect?
#
# 5. Exercises (Day 5)
# - Change learning_rate to 0.001 and 0.1. Observe convergence.
# - Add a new data point [5,5] and rerun GD.
# - Plot MSE vs iterations. Q: What do you observe?
# - Q: Why do we multiply by 2/m in the gradient formula?

# import random

X = np.array([[1],[2],[3],[4]])
Y = np.array([[1],[2],[2],[4]])

X_b = np.c_[np.ones((4,1)),X]
print(f"X_b = {X_b}")


theta = np.random.randn(2,1)
learning_rate = 0.01
n_iterations = 1000
m = len(Y)

for iteration in range(n_iterations):
  gradient = (2/m) * X_b.T @ (X_b @ theta - Y)
  theta = theta - learning_rate * gradient

print(f"theta found by gradient descent : {theta}")  # Bias = 0, slope = 1

# price=1⋅size + 0 as m = 1 and b(intercept) = 0
size = 5
X_new = np.array([1, size])
Y_pred = X_new @ theta
print(f"Predicted Price for size {size} => {Y_pred}")

