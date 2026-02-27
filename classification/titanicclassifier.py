import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

data_path = Path("../Datasets/Spaceship Titanic")
train_df = pd.read_csv(data_path/"train.csv")
test_df = pd.read_csv(data_path/"test.csv")

def preprocess_data(X):
    # Categorize numerical and categorical features
    numeric_features = X.select_dtypes(include=["number"]).columns
    categorical_features = X.select_dtypes(include=["object", "category"]).columns
    # X[categorical_features] = X[categorical_features].astype(str)

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    bool_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    return preprocessor

def engineer_features(df):
    # Always work on a copy to avoid SettingWithCopy warnings
    df = df.copy()
    
    # Extract Cabin features based on the data dictionary (deck/num/side)
    # Missing values here just become NaNs, which our pipeline imputer will handle later!
    df[['CabinDeck', 'CabinNum', 'CabinSide']] = df['Cabin'].str.split('/', expand=True)
    df['CabinNum'] = pd.to_numeric(df['CabinNum'], errors='coerce')
    
    # Feature Engineering: Total Spending
    # People in CryoSleep can't spend money. Grouping these helps the model find patterns.
    spend_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df['TotalSpend'] = df[spend_cols].sum(axis=1)
    
    # Convert booleans to strings so the categorical pipeline handles them consistently
    df['CryoSleep'] = df['CryoSleep'].astype(str)
    df['VIP'] = df['VIP'].astype(str)
    
    # Drop columns that are too unique (Name) or have been replaced (Cabin)
    # We DO NOT drop PassengerId yet, because we need it for the submission file!
    df = df.drop(columns=['Cabin', 'Name'], errors='ignore')
    
    return df

# Convert the target variabale values to target encoded values
y = train_df['Transported'].astype(int)
X = train_df.drop(columns=['Transported'])
X_train_final = engineer_features(X)
X_test_final = engineer_features(test_df)

# Split the dataset
X_train, X_val, y_train, y_val = train_test_split(X_train_final, y, test_size=0.2, random_state=42)

# Preprocessor pipeline
preprocessor = preprocess_data(X_train)
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    ))
])

# Training the model
print("Training started..")
model_pipeline.fit(X_train, y_train)

# Validating mode
print("Validating the model..")
val_preds = model_pipeline.predict(X_val)
validation_accuracy = accuracy_score(y_val, val_preds)
print(f"Validation accuracy: {validation_accuracy}")

# Generating predictions
print("Generating predictions for test.csv..")
test_preds = model_pipeline.predict(X_test_final)
submission_results = pd.DataFrame({
    "PassengerId": X_test_final['PassengerId'],
    "Transported": test_preds.astype(bool)
})
submission_results.to_csv("titanic_classifier_results.csv", index=False)
print("Success!")