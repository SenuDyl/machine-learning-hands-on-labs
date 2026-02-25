import pandas as pd
import numpy as np

permits_path = "../Datasets/San Francisco Building Permits/Building_Permits.csv"
permits_data = pd.read_csv(permits_path)
# print(permits_data.head())

# Print total cells with missing values of the entire dataset
total_missing_values = permits_data.isnull().sum().sum()
# print(total_missing_values)
data_shape = permits_data.shape
# print(data_shape)
total_cells = np.prod(data_shape)
# print(total_cells)
percentage_missing = (total_missing_values/total_cells)*100
# print(percentage_missing)

# Print all the columns with missing values and missing value counts
missing_counts = permits_data.isnull().sum().sort_values(ascending=False)
# print(missing_counts)
missing_cols = missing_counts[missing_counts>0]
# print(missing_cols)

# Drop rows with missing values
permits_data_with_na_dropped_rows = permits_data.dropna()
print(permits_data_with_na_dropped_rows.shape)

# Drop columns with missing values
permits_data_with_na_dropped_columns = permits_data.dropna(axis=1)
print(permits_data_with_na_dropped_columns.shape)

# Filling missing values automatically - imputing
# Replacing all the NaN's in the sf_permits data with the one that comes directly after it and then replacing any remaining NaN's with 0. 
permits_data_with_na_imputed = permits_data.bfill(axis=0).fillna(0)
print(permits_data_with_na_imputed.shape)