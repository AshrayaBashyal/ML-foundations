# Project:

# ---------------- Dataset ----------------
# Day | Open   | High   | Low    | Close
# 1   | 100    | 105    | 98     | 102
# 2   | NaN    | 108    | 101    | 107
# 3   | 110    | NaN    | 106    | 108
# 4   | 115    | 120    | NaN    | 117
# 5   | 118    | 123    | 115    | NaN

# Replace NaNs with:
# Mean 
# Median
# A fixed value (e.g., 0 or 999 for “missing”)

# Normalize with:
# Min-Max scaling 
# Z-score scaling 

# Add new features:
# Daily range = (max - min per day)
# Daily variance

# (Conceptual)
# Why do we normalize data before ML?
# When would median replacement be better than mean replacement?


import numpy as np

# ---------------- Dataset ----------------
data = np.array([
    [100, 105, 98, 102],
    [np.nan, 108, 101, 107],
    [110, np.nan, 106, 108],
    [115, 120, np.nan, 117],
    [118, 123, 115, np.nan]
])

# ---------------- Handle Missing Values ----------------
def replace_nan_with_mean(data):
    """Replace NaNs with column-wise mean."""
    col_means = np.nanmean(data, axis=0)
    return np.where(np.isnan(data), col_means, data)

def replace_nan_with_median(data):
    """Replace NaNs with column-wise median."""
    col_medians = np.nanmedian(data, axis=0)
    return np.where(np.isnan(data), col_medians, data)

def replace_nan_with_fixed(data, value=0):
    """Replace NaNs with a fixed value."""
    return np.where(np.isnan(data), value, data)


# ---------------- Feature Normalization ----------------
def min_max_normalize(data):
    """Scale each column to [0, 1] using: (x - min) / (max - min)."""
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    return (data - min_vals) / (max_vals - min_vals)

def z_score_normalize(data):
    """Standardize columns (mean=0, std=1): z = (x - mean)/std."""
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)
    return (data - means) / stds


# ---------------- Feature Engineering ----------------
def daily_range(data):
    """Daily Range = High - Low (column 1 - column 2)."""
    return data[:, 1] - data[:, 2]

def daily_variance(data):
    """Row-wise variance across all columns."""
    return np.var(data, axis=1)


# ---------------- Example Usage ----------------
# Handle missing values
data_mean = replace_nan_with_mean(data)
data_median = replace_nan_with_median(data)
data_fixed = replace_nan_with_fixed(data, value=0)

# Normalize features
data_minmax = min_max_normalize(data_mean)
data_zscore = z_score_normalize(data_mean)

# Add new features
range_feature = daily_range(data_mean)
variance_feature = daily_variance(data_mean)

# ---------------- Print Results ----------------
print("Data (NaN -> MEAN):\n", data_mean)
print("\nData after Min-Max Normalization:\n", data_minmax)
print("\nData after Z-Score Normalization:\n", data_zscore)
print("\nDaily Range (High-Low):\n", range_feature)
print("\nDaily Variance:\n", variance_feature)
