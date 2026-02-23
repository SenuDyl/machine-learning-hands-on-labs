import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

home_data_path = "./Home data for ML course/train.csv"
home_data = pd.read_csv(home_data_path)

features = ["OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF", "FullBath", "YearBuilt"]
X = home_data[features]

y = home_data["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

randomforest_model = RandomForestRegressor(random_state=1)
randomforest_model.fit(X_train, y_train)

y_pred = randomforest_model.predict(X_test)

rf_error = mean_absolute_error(y_test, y_pred)

print(f"Validation MAE for random forest model: {format(round(rf_error, 2))}")

output = pd.DataFrame({
    "Actual values": y_test,
    "Predicted values": y_pred
})

output_path = "out/rf-iow_house.csv"

output.to_csv(output_path, index=False)