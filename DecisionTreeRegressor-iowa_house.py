import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

home_data_path = "./Datasets/Home data for ML course/train.csv"
home_data = pd.read_csv(home_data_path)
# print(home_data.head())

# Print the column names to find the target variable and features
# print(home_data.columns)

# Creating the list of features
features = ["OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF", "FullBath", "YearBuilt"]

X = home_data[features]

# print(X.describe())
# print(X.head())

# Define the target variable
y = home_data["SalePrice"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# Create the model
iowa_model = DecisionTreeRegressor(random_state=1)

# Fit the model
iowa_model.fit(X_train, y_train)

# Make predictions
val_predictions = iowa_model.predict(X_test)
# print(val_predictions)

# Compare predictions to actual values
actual_values = y_test.head()
comparison = pd.DataFrame({
    "Actual": actual_values,
    "Predicted": val_predictions[:5]
})

# print(comparison)

# Calculate the mean absolute error in the validation set
val_error = mean_absolute_error(y_test, val_predictions)
print("Validation MAE: ", val_error)

# Comparing against a baseline model
baseline_prediction = y_train.mean()
baseline_predictions = [baseline_prediction] * len(y_test)  
baseline_error = mean_absolute_error(y_test, baseline_predictions)
print("Baseline MAE: ", baseline_error)

# Validation MAE:  26435.205479452055
# Baseline MAE:  59219.61825483206
# This shows that the Decision Tree Regressor model performs significantly better than the baseline model, which simply predicts the mean sale price for all houses.

# Hyperparameter tuning: Adjusting the max_leaf_nodes parameter to find the optimal number of leaf nodes for the decision tree

def get_mae(leaf_node_no):
    model = DecisionTreeRegressor(max_leaf_nodes=leaf_node_no, random_state=1)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    return mae

leaf_node_counts = [5, 25, 50, 100, 250, 500]
mae_values = []

for leaf_count in range(len(leaf_node_counts)):
    mae_values.append(get_mae(leaf_node_counts[leaf_count]))

print(mae_values)

best_tree_size = leaf_node_counts[mae_values.index(min(mae_values))]
print(best_tree_size)