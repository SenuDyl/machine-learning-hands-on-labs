import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

# 1. Load the data
data_dir = Path("./Datasets/Predicting Heart Disease")
train = pd.read_csv(data_dir/'train.csv')
test = pd.read_csv(data_dir/'test.csv')

# Add new features
def add_new_features(df):
    # a. BP/Age Ratio: High blood pressure is more significant at younger ages
    df['BP_Age_Ratio'] = df['BP'] / (df['Age'] + 1)
    
    # b. Cholesterol/Age Ratio: Cumulative exposure to high cholesterol
    df['Chol_Age_Ratio'] = df['Cholesterol'] / (df['Age'] + 1)
    
    # c. Cardiovascular Risk Factor: Interaction between BP and Cholesterol
    df['CV_Risk_Score'] = df['BP'] * df['Cholesterol']
    
    # d. Heart Rate Efficiency: Interaction between Age and Max HR
    df['HR_Efficiency'] = df['Max HR'] / (220 - df['Age'])
    
    # e. Exercise Impact: Interaction between Angina and ST Depression
    df['Exercise_Stress'] = df['Exercise angina'] * df['ST depression']
    
    return df

train = add_new_features(train)
test = add_new_features(test)

# Identify features and target
target = 'Heart Disease'
drop_cols = ['id', target]
features = [c for c in train.columns if c not in drop_cols]

# 2. Preprocessing: Handle Categorical Variables
cat_features = train[features].select_dtypes(include=['object']).columns

for col in cat_features:
    le = LabelEncoder()
    full_data = pd.concat([train[col], test[col]], axis=0).astype(str)
    le.fit(full_data)
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

# Fix Target Labels
target_map = {'Absence': 0, 'Presence': 1}
train[target] = train[target].map(target_map)

X = train[features]
y = train[target]
X_test = test[features]

# 3. Model Configuration
xgb_params = {
    'n_estimators': 1500,       # Increased slightly
    'learning_rate': 0.03,       # Lowered for better convergence
    'max_depth': 5,              # Slightly shallower to prevent overfitting
    'subsample': 0.85,
    'colsample_bytree': 0.7,     # Forces model to use different feature subsets
    'n_jobs': -1,
    'eval_metric': 'auc',
    'objective': 'binary:logistic',
    'random_state': 42,
    'tree_method': 'hist',
}

# 4. Cross-Validation Loop
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))

print("Starting Cross-Validation with Engineered Features...")

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = XGBClassifier(**xgb_params)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    test_preds += model.predict_proba(X_test)[:, 1] / skf.n_splits
    
    fold_auc = roc_auc_score(y_val, oof_preds[val_idx])
    print(f"Fold {fold+1} ROC-AUC: {fold_auc:.5f}")

# 5. Final Evaluation
total_auc = roc_auc_score(y, oof_preds)
print(f"\nOverall Out-of-Fold ROC-AUC: {total_auc:.5f}")

# 6. Create Submission File
submission = pd.DataFrame({
    'id': test['id'],
    'Heart Disease': test_preds
})

submission.to_csv('out/heart-disease-submission_v2.csv', index=False)
print("\nSuccess! 'submission_v2.csv' is ready.")