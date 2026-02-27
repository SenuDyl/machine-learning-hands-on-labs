# Feature Engineering - MI scores
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

df = pd.read_csv("../Datasets/FE Course Data/ames.csv")

def make_mi_scores(X, y):
    X=X.copy()
    for col_name in X.select_dtypes(["object", "category"]):
        X[col_name], _ = X[col_name].factorize()
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name="MI-scores", index = X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

X = df.copy()
y = X.pop('SalePrice')
mi_scores = make_mi_scores(X, y)
print(mi_scores)