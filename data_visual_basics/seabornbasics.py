import seaborn as sns
import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np

# Example dataset
df = sns.load_dataset("tips")  # Seaborn comes with built-in datasets
print(df.head())

# Themes & Styles
sns.set_style("whitegrid")  # other: "darkgrid", "white", "ticks", "dark"

# Basic Plot Types:

# Scatter Plot:
# Shows relationship between two numeric variables.
sns.scatterplot(x="total_bill", y="tip", data=df)
plt.show()

# Line Plot:
# Shows trends across a variable.
sns.lineplot(x="size", y="tip", data=df)
plt.show()

# Histogram / Distribution Plot:
# Shows distribution of a single variable.
sns.histplot(df["total_bill"], bins=20, kde=True)  # kde=True adds smooth curve
plt.show()

# Box Plot:
# Good for spotting outliers and comparing distributions across categories.
sns.boxplot(x="day", y="total_bill", data=df)
plt.show()

# Violin Plot:
# Like boxplot, but shows distribution density too.
sns.violinplot(x="day", y="total_bill", data=df)
plt.show()

# Heatmap:
# Visualizes correlation between numeric variables.
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.show()


