import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- Dataset ----------------
data = {
    "age": [63, 37, 41, 56, 57, 57, 56, 44, 52, 57],
    "sex": ["Male", "Female", "Male", "Male", "Female", "Male", "Female", "Male", "Male", "Female"],
    "cholesterol": [233, 250, 204, 236, 354, 192, 294, 263, 199, 168],
    "resting_bp": [145, 130, 130, 120, 130, 120, 140, 120, 172, 150],
    "max_heart_rate": [150, 187, 172, 178, 163, 148, 153, 173, 162, 174],
    "exercise_angina": ["No", "No", "Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No"],
    "st_depression": [2.3, 3.5, 1.4, 0.8, 1.2, 0.6, 1.8, 0.0, 1.5, 0.4],
    "disease": [1, 0, 0, 1, 1, 0, 1, 0, 1, 0]  # 1 = diseased, 0 = healthy
}

df = pd.DataFrame(data)

# =======================================================
# Mini Tasks:
# =======================================================

# Q1. Plot histograms of age and cholesterol using Matplotlib.
#     (Learn: plt.hist, labels, titles, subplots)
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.hist(df["age"], bins=5, color="skyblue", edgecolor="black")
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")

plt.subplot(1, 2, 2)
plt.hist(df["cholesterol"], bins=5, color="lightgreen", edgecolor="black")
plt.title("Cholesterol Distribution")
plt.xlabel("Cholesterol")
plt.ylabel("Count")

plt.tight_layout()
# plt.show()


# Q2. Create a scatter plot of age vs cholesterol, color by disease.
#     (Learn: sns.scatterplot, hue, axis labels, titles)
plt.figure(figsize=(6, 4))
sns.scatterplot(x="age", y="cholesterol", hue="disease", data=df, palette="Set1")
plt.title("Age vs Cholesterol (Colored by Disease)")
plt.xlabel("Age")
plt.ylabel("Cholesterol")
# plt.show()


# Q3. Compute correlation matrix and plot heatmap.
#     (Learn: df.corr, sns.heatmap, annot=True)
corr = df.corr(numeric_only=True)

plt.figure(figsize=(6, 4))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
# plt.show()


# Q4. Bonus: Plot boxplots of cholesterol for diseased vs non-diseased patients.
#     (Learn: sns.boxplot, x/y usage, category comparison)
plt.figure(figsize=(6, 4))
sns.boxplot(x="disease", y="cholesterol", data=df, palette="pastel")
plt.title("Cholesterol Levels: Diseased vs Healthy")
plt.xlabel("Disease (0 = Healthy, 1 = Diseased)")
plt.ylabel("Cholesterol")
plt.show()
