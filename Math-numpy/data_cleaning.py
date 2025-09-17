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
  col_mdians = np.nanmedian(X, axis=1)
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


data = np.array([
    [30, 32, np.nan, np.nan],  # Day 1
    [31, np.nan, 35, np.nan],  # Day 2
    [np.nan, 34, 36, np.nan],  # Day 3
    [33, 35, 37, np.nan]       # Day 4
])

print(missing_summary(data))
print(impute_column_mean(data))
print(impute_column_mean(data))
print(impute_constant(data))

