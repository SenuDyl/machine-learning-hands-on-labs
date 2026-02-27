import pandas as pd
from sklearn.feature_selection import mutual_info_regression

def make_mi_scores(X, y):
    X = X.copy()

    # Encode categorials
    for colname in X.select_dtypes(["object", "category"]):
        X[colname], _ = X[colname].factorize()

    # Impute missing values because mutual_info_regression can't handle them
    X = X.fillna(0)

    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=42)
    mi_scores = pd.Series(mi_scores, index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

def drop_uninformative(df, mi_scores):
    return df.loc[:, mi_scores>0.0]