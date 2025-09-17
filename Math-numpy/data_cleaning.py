import numpy as np

def missing_summary(X):
  """Return total and per-column missing counts and fraction."""
  total_missing = np.isnan(X).sum()
  per_col = np.isnan(X).sum(axis = 0)
  fraction_per_col =  per_col / X.shape[0]  
  return {
    "total missing" : int(total_missing),
    "per_column missing" : per_col,
    "fraction per_column" : fraction_per_col
  }

def impute_column_mean(X):
  """
  Replace NaN in each column by the column mean (ignoring NaN).
  Returns a new array (float dtype).
  """  
  X = X.astype(float, copy=True)
  col_means = np.nanmean(X, axis=0)
  inds = np.where(np.isnan(X))
  nan_cols = np.isnan(col_means)
  if np.any(nan_cols):
    col_means[nan_cols] = 0
  X[inds] = np.take(col_means, inds[1])  
  return X
  # return{
    # "col_means" : col_means,
    # "inds" : inds,
    # "nan_cols" : nan_cols,
    # "Data" : X,
    # "inds[0]" : inds[0],
    # "inds[1]" : inds[1],
    # "X[inds]" : X[inds]
  # }

def impute_column_median(X):
  """Replace NaN in each column by the column median (ignoring NaN)."""
  X = X.astype(float, copy=True)
  col_mdians = np.nanmedian(X, axis=0)
  inds = np.where(np.isnan(X))
  nan_cols = np.isnan(col_mdians)
  if np.any(nan_cols):
    col_mdians[nan_cols] = 0
  inds[X] = np.take(col_mdians, inds[1])  
  return X
    

def impute_constant(X, value=0.0):
    """Replace all NaNs with a constant value."""
    X = X.astype(float, copy=True)
    X[np.isnan(X)] = value
    return X

# ---------- Drop rows/columns ----------
def drop_rows_with_nan(X, how='any'):
    """If how='any' drop row if any NaN; if how='all' drop if all NaN."""
    if how == 'any':
        mask = ~np.isnan(X).any(axis=1)
    else:
        mask = ~np.isnan(X).all(axis=1)
    return X[mask]

def drop_cols_with_nan(X, thresh=0.5):
    """
    Drop columns with fraction of NaN >= thresh.
    thresh between 0 and 1 (e.g., 0.5 -> drop columns with >=50% missing).
    """
    frac = np.isnan(X).sum(axis=0) / X.shape[0]
    keep = frac < thresh
    return X[:, keep], keep  # return mask of kept columns

# # ------------Normalize Features-----Min-Max Normalization-----Z-Score Normalization------------

# Normalization = rescaling features for fair contribution.
# Min-Max: [0,1], sensitive to outliers, preserves distribution.
# Z-Score: mean=0, std=1, useful for Gaussian-like data.
# Choice depends on algorithm and dataset.

def min_max_normalize(data):
    """
    Min-Max Normalization:
    Scales each feature (column) to [0, 1] using:
        x' = (x - min) / (max - min)
    """
    min_vals = data.min(axis=0)
    max_vals = data.max(axis=0)
    return (data - min_vals) / (max_vals - min_vals)

def z_score_normalize(data):
    """
    Z-Score Normalization:
    Standardizes each feature (column) to mean=0, std=1 using:
        z = (x - μ) / σ
    """
    mean_vals = data.mean(axis=0)
    std_vals = data.std(axis=0)
    return (data - mean_vals) / std_vals

def add_daily_average(data):
    """
    Extra Feature - Daily Average:
    For each row (day), compute average of all columns (sensors)
    and append it as a new column.
    """
    daily_avg = data.mean(axis=1).reshape(-1, 1)
    return np.hstack([data, daily_avg])

data = np.array([
    [30, 32, np.nan, np.nan],  # Day 1
    [31, np.nan, 35, np.nan],  # Day 2
    [np.nan, 34, 36, np.nan],  # Day 3
    [33, 35, 37, np.nan]       # Day 4
])

print("Missing Summary:\n", missing_summary(data))
print("Impute Column Mean:\n", impute_column_mean(data))
print("Impute Column Median:\n", impute_column_median(data))
print("Impute Column Constants:\n", impute_constant(data))
print("Min-Max Normalized:\n", min_max_normalize(data))
print("Z-Score Normalized:\n", z_score_normalize(data))
print("With Daily Average:\n", add_daily_average(data))

