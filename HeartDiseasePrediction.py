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

# Identify features and target
target = 'Heart Disease'
drop_cols = ['id', target]
features = [c for c in train.columns if c not in drop_cols]

# 2. Preprocessing: Handle Categorical Variables
# Synthetic datasets often have 'Object' types. Let's encode them.
cat_features = train[features].select_dtypes(include=['object']).columns

for col in cat_features:
    le = LabelEncoder()
    # Fit on both train and test to ensure all labels are captured
    full_data = pd.concat([train[col], test[col]], axis=0).astype(str)
    le.fit(full_data)
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

target_map = {'Absence': 0, 'Presence': 1}
train[target] = train[target].map(target_map)
X = train[features]
y = train[target]
X_test = test[features]

# 3. Model Configuration
# These are "safe" hyperparameters for synthetic tabular data
xgb_params = {
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'n_jobs': -1,
    'eval_metric': 'auc',
    'objective': 'binary:logistic',
    'random_state': 42,
    'tree_method': 'hist', # Faster training
}

# 4. Cross-Validation Loop (5-Folds)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))

print("Starting Cross-Validation...")

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = XGBClassifier(**xgb_params)
    
    # Use early stopping to prevent overfitting
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Predict probabilities for the positive class (Heart Disease)
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

submission.to_csv('out/heart-disease-submission.csv', index=False)
print("\nSubmission file 'submission.csv' has been saved!")