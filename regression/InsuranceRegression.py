import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

data_path = Path("../Datasets/Insurance")
insurance_data = pd.read_csv(data_path/'insurance.csv')

# Feature engineering - It acually increased the error
# insurance_data['smoker_and_obese'] = ((insurance_data['bmi'] >= 30) & (insurance_data['smoker'] == 'yes')).astype(str)

y = insurance_data['charges']
X = insurance_data.drop(columns=['charges'])

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

numerical_features = X.select_dtypes(include="number").columns
categorical_features = X.select_dtypes(include=["object", "category"]).columns

numeric_transformer = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numerical_features),
    ('cat', categorical_transformer, categorical_features)
])

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(random_state=42))
])

# Grid search
param_grid = {
    'model__n_estimators': [50, 100, 200],
    'model__max_depth': [3, 5, 7, None],
    'model__min_samples_split': [2, 5, 10]
}

print("Starting Grid Search to find the best hyperparameters...")
grid_search = GridSearchCV(
    estimator=model_pipeline, 
    param_grid=param_grid, 
    cv=5, # 5-Fold Cross Validation
    scoring='neg_root_mean_squared_error',
    n_jobs=-1 # Uses all available CPU cores to speed up training
)

grid_search.fit(train_X, train_y)

# 8. Evaluate the Best Model
best_model = grid_search.best_estimator_
print(f"\nBest Hyperparameters Found: {grid_search.best_params_}")

# model_pipeline.fit(train_X,train_y)
y_pred = best_model.predict(test_X)
rmse = np.sqrt(mean_squared_error(test_y, y_pred))
r2 = r2_score(test_y, y_pred)

print(f"Test set root mean square error: {rmse:2f}")
print(f"Test set r2 score: {r2}")

results = pd.DataFrame({
    "Actual Value": test_y,
    "Predicted Value": y_pred
})

results.to_csv("InsuranceRegression_results.csv")
