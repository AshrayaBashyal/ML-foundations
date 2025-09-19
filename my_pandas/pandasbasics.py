import pandas as pd

# Load the patients.csv file from the same directory
df = pd.read_csv(r"C:\Users\Ashraya Bashyal\Desktop\ML\my_pandas\patients.csv")


# First Look at the Data
print("\nFirst 5 rows:\n")
print(df.head())
print("\n\nLast five rows:\n")
print(df.tail())
print("\n\nRows, Columns:\n")
print(df.shape)
print("\n\ncolumn types, non-null counts:\n")
print(df.info())
print("\n\nsummary stats (mean, min, max, etc.):\n")
print(df.describe())

# Selecting Columns
print(df["Age"])
print(df[["Gender", "Age"]])


# Selecting Rows

# .iloc[] → index-based (numbers)
# df.iloc[row_selection, column_selection]
# df.iloc[i] → selects row i as a Series
# df.iloc[[i]] → selects row i as a DataFrame

print(df.iloc[0])
print(df.iloc[0:3, 0:5])
df.iloc[[0, 3], [1, 2]]  # rows 0 and 3, columns 1 and 2

# .loc[] → label-based (names/conditions)
# df.loc[row_selection, column_selection]
print(df.loc[0])
print(df.loc[df["Age"]>60, ["Age", "Gender", "Condition", "Outcome"]])

df_copy = df.copy()
df_copy.loc[df_copy["Age"]>60, "Outcome"] = "Critical"  # changes outcomes of all patients over 60 to Critical
print(df_copy.loc[df_copy["Age"]>60, "Outcome"])

# Filtering Rows (Boolean Indexing)
older_patients = df[df["Age"] > 50]
print(older_patients)

# male_over_50 = df.query('Age > 50 and Gender == "Male"')
male_over_50 = df[(df["Age"]>50) & (df["Gender"]=="Male")]
print(male_over_50)


# Mini-Task 

# Load a CSV dataset (I suggest heart disease dataset: heart.csv if you have it).
# Show:
# First 10 rows.
# Total rows and columns.
# Patients older than 50.
# Columns: age, sex, and cholesterol.

df = pd.read_csv(r"C:\Users\Ashraya Bashyal\Desktop\ML\my_pandas\heart.csv")

print("\nFirst 10 rows:\n\n")
print(df.head(10))
print("\nTotal Rows and Columns:\n\n")
print(df.shape)
print("\nPatients older than 50:\n\n")
print(df[df["age"]>50])
print("\nColumns: age, sex and cholesterol:\n\n")
print(df.loc[:,["age", "sex", "cholesterol"]])




# Handling Missing Values, Duplicates & Grouping

# Handling Missing Data (NaN)
import numpy as np

data = {
    "age": [25, 30, 35, np.nan, 40],
    "cholesterol": [200, 210, np.nan, 220, 230],
    "sex": ["male", "female", "female", np.nan, "female"]  
}

df = pd.DataFrame(data)
print("\n\nOriginal Data:\n", df)

# 1.1 Detect Missing Values
print("\n\nMissing Values:\n")
print(df.isna())
print(df.isna().sum())

# # 1.2 Drop Missing Values
print("\n\nDrop Missing Values:\n")
df_drop = df.dropna()
print(df_drop)
# 1.3 Fill Missing Values
print("\n\nFill Missing Values:\n")

new_df = df.copy()

new_df["age"] = new_df["age"].fillna(df["age"].mean())
new_df["cholesterol"] = new_df["cholesterol"].fillna(df["cholesterol"].mean())

# Why Not Mean for Categorical Data?
# If a column is "sex" = Male / Female / NaN → taking the mean makes no sense, because they are not numbers.
# Instead, we use the mode:
# The mode = most frequent (most common) value in that column.
# Example: If "sex" = [Male, Male, Female, NaN, Male] → mode = "Male".

new_df["sex"] = new_df["sex"].fillna(df["sex"].mode()[0])
print(new_df)


# 2. Removing Duplicates

# df.iloc[[0]] ensures the first row is a DataFrame, so concat works.
# ignore_index=True reindexes the resulting DataFrame.

print("\n\nAdding Duplicates:\n")
df_dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
print(df_dup)

print("\n\nRemoving Duplicates:\n")
df_no_dup = df_dup.drop_duplicates()
print(df_no_dup)


# 3. Grouping Data (groupby): Grouping helps summarize data.
# Can also do count(), max(), min()
# Useful for exploratory data analysis (EDA)
  
print("\n\nAverage cholesterol by sex:\n")
grouped = df.groupby("sex")["cholesterol"].mean()
print(grouped)

# Mini Task:
# Load your heart disease dataset.
# Count missing values per column.
# Fill missing age with median, sex with mode and cholesterol with mean.
# Remove duplicates if any.
# Find average cholesterol per sex.
# Bonus: Find average cholesterol per age group (df['age'] // 10 * 10)

df = pd.read_csv("C:/Users/Ashraya Bashyal/Desktop/ML/my_pandas/heart.csv")

Missing = df.isna().sum()
print("\n\nMissing values per column:\n",Missing)

df["age"] = df["age"].fillna(df["age"].median())
df["sex"] = df["sex"].fillna(df["sex"].mode()[0])
df["cholesterol"] = df["cholesterol"].fillna(df["cholesterol"].mean())
print(df.loc[:,["age", "sex", "cholesterol"]])

df = df.drop_duplicates()
print("\n\nRemoved duplicates if any:\n",df)

sex_grouped = df.groupby("sex")["cholesterol"].mean()
print("\n\nAverage cholesterol per sex:\n", sex_grouped)

df["age_group"] = (df["age"]//10)*10
age_grouped = df.groupby("age_group")["cholesterol"].mean()
print("\n\nAverage cholesterol per age group:\n", age_grouped)