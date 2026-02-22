import pandas as pd
from sklearn.tree import DecisionTreeRegressor

home_data_path = "./Home data for ML course/train.csv"
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

# Create the model
iowa_model = DecisionTreeRegressor(random_state=1)

# Fit the model
iowa_model.fit(X, y)

# Make predictions
predictions = iowa_model.predict(X)
print(predictions)

# Compare predictions to actual values
actual_values = y.head()
comparison = pd.DataFrame({
    "Actual": actual_values,
    "Predicted": predictions[:5]
})

print(comparison)