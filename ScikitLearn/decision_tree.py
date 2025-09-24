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