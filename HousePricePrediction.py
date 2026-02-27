# Kaggle competition - Feature Engineering
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from utility import make_mi_scores, drop_uninformative

# Data preprocessing
# 1. Data cleaning
# 2. Data encoding
# 3. Data imputing

# Data cleaning - Make categorical labels consistent, rename columns, fix corrupt data
def clean(df):
    cat_cols = df.select_dtypes(include="object").columns
    num_cols = df.select_dtypes(exclude="object").columns
    for col in cat_cols:
        df[col] = df[col].str.strip()
        # df[col] = df[col].str.lower()
        # print(f"\n{col}")
        # print(df[col].value_counts())
    df['MSZoning'] = df['MSZoning'].replace({'c (all)': 'c'})
    df['Exterior2nd'] = df['Exterior2nd'].replace({
        'CmentBd': 'CemntBd',
        'Wd Shng': 'Wd Sdng',
        'Brk Cmn': 'BrkComm'
    })

    df['GarageYrBlt'] = df['GarageYrBlt'].where(df.GarageYrBlt<=2010, df.YearBuilt)

    # Rename the columsn beginning with a number
    df.rename(columns = {
        "1stFlrSF": "FirstFlrSF",
        "2ndFlrSF": "SecondFlrSF",
        "3SsnPorch": "Threeseasonporch",
    }, inplace=True)

    return df

# Data encoding - Encode categorical data (both nominal and ordinal) so the model identify them correctly
# The nominative (unordered) categorical features
features_nom = ["MSSubClass", "MSZoning", "Street", "Alley", "LandContour", "LotConfig", "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle", "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType", "Foundation", "Heating", "CentralAir", "GarageType", "MiscFeature", "SaleType", "SaleCondition"]

# The ordinal (ordered) categorical features 

# Pandas calls the categories "levels"
five_levels = ["Po", "Fa", "TA", "Gd", "Ex"]
ten_levels = list(range(0, 11))
ordered_levels = {
    "OverallQual": ten_levels,
    "OverallCond": ten_levels,
    "ExterQual": five_levels,
    "ExterCond": five_levels,
    "BsmtQual": five_levels,
    "BsmtCond": five_levels,
    "HeatingQC": five_levels,
    "KitchenQual": five_levels,
    "FireplaceQu": five_levels,
    "GarageQual": five_levels,
    "GarageCond": five_levels,
    "PoolQC": five_levels,
    "LotShape": ["Reg", "IR1", "IR2", "IR3"],
    "LandSlope": ["Sev", "Mod", "Gtl"],
    "BsmtExposure": ["No", "Mn", "Av", "Gd"],
    "BsmtFinType1": ["Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "BsmtFinType2": ["Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "Functional": ["Sal", "Sev", "Maj1", "Maj2", "Mod", "Min2", "Min1", "Typ"],
    "GarageFinish": ["Unf", "RFn", "Fin"],
    "PavedDrive": ["N", "P", "Y"],
    "Utilities": ["NoSeWa", "NoSewr", "AllPub"],
    "CentralAir": ["N", "Y"],
    "Electrical": ["Mix", "FuseP", "FuseF", "FuseA", "SBrkr"],
    "Fence": ["MnWw", "GdWo", "MnPrv", "GdPrv"],
}

ordered_levels = {k: ["None"] + v for k, v in ordered_levels.items()}

def encode_and_impute(df):
    features_nom_existing = [c for c in features_nom if c in df.columns]

    # 1. Nominal (unordered) pipeline
    nominal_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # 2. Ordinal (ordered) pipeline
    ordinal_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
        ("ordinal", OrdinalEncoder(dtype=object))  # dtype=object allows mixed strings
    ])

    # Make sure all ordinal columns are strings
    ordinal_cols = [c for c in ordered_levels.keys() if c in df.columns]
    df[ordinal_cols] = df[ordinal_cols].astype(str)

    # Impute numerical values
    numeric_cols = [c for c in df.select_dtypes(include="number").columns if c not in ordinal_cols]
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("nominal", nominal_transformer, features_nom_existing),
        ("ordinal", ordinal_transformer, ordinal_cols),
        ("numeric", numeric_transformer, numeric_cols)
    ])

    preprocessed = preprocessor.fit_transform(df)
    # preprocessed_dense = preprocessed.toarray()  # Converts sparse to dense numpy array
    df_encoded = pd.DataFrame(preprocessed, index=df.index)
    return df_encoded


def load_data():
    data_dir = Path("./Datasets/House Prices - Advanced Regression")
    df_train = pd.read_csv(data_dir/"train.csv", index_col="Id")
    df_test = pd.read_csv(data_dir/"test.csv", index_col="Id")

    # Merge the splits so we can process them together
    df = pd.concat([df_train, df_test])

    # Preprocess
    df = clean(df)

    # Reform splits
    df_train = df.loc[df_train.index, :]
    df_test = df.loc[df_test.index, :]
    return df_train, df_test

def score_dataset(X_train, y_train):
    model = XGBRegressor(
        n_estimators = 100,
        learning_rate = 0.1,
        max_depth = 5,
        random_state = 42
    )

    log_y = np.log(y_train + 1e-9)
    score = cross_val_score(
        model, X_train, log_y, cv=5, scoring="neg_mean_squared_error"
    )
    score = -1 * score.mean()
    score = np.sqrt(score)
    return score

# Create features
def mathematical_transforms(df):
    X = pd.DataFrame()
    X["LivLotRatio"] = df.GrLivArea / (df.LotArea + 1)
    X["Spaciousness"] = (df.FirstFlrSF + df.SecondFlrSF) / (df.TotRmsAbvGrd + 1)
    return X

def interactions(df):
    X = pd.get_dummies(df.BldgType, prefix="Bldg")
    X = X.mul(df.GrLivArea, axis=0)
    return X

def counts(df):
    X = pd.DataFrame()
    porch_features = ["WoodDeckSF",
        "OpenPorchSF",
        "EnclosedPorch",
        "Threeseasonporch",
        "ScreenPorch"]
    
    X["PorchTypes"] = df[porch_features].gt(0.0).sum(axis=1)
    return X

def break_down(df):
    X = pd.DataFrame()
    X["MSClass"] = df.MSSubClass.str.split("_", n=1, expand=True)[0]
    return X

def group_transform(df):
    X = pd.DataFrame()
    X["MedNhbdArea"] = df.groupby("Neighborhood")["GrLivArea"].transform("median")
    return X

# Feature engineering using KMeans clustering
cluster_features = [
    "LotArea",
    "TotalBsmtSF",
    "FirstFlrSF",
    "SecondFlrSF",
    "GrLivArea",
]

# Define cluster labels
def cluster_labels(df, features, n_clusters=20):
    X = df.copy()
    X_scaled = X.loc[:, features]
    X_scaled = (X_scaled - X_scaled.mean(axis=0))/X_scaled.std(axis=0)
    kMeans = KMeans(n_clusters=n_clusters, n_init=50, random_state=0)
    X_new = pd.DataFrame()
    X_new['Cluster'] = kMeans.fit_predict(X_scaled)
    return X_new

# Define cluster distance
def cluster_distance(df, features, n_clusters=20):
    X = df.copy()
    X_scaled = X.loc[:, features]
    X_scaled = (X_scaled - X_scaled.mean(axis=0))/X_scaled.std(axis=0)
    kMeans = KMeans(n_clusters=n_clusters, n_init=50, random_state=0)
    X_cd = kMeans.fit_transform(X_scaled)
    X_cd = pd.DataFrame(
        X_cd, columns=[f"Centroid_{i}" for i in range(X_cd.shape[1])]
    )
    return X_cd

def select_features_to_apply_pca(X, y, top_n=5):
    # Select only numeric features because correlation can be calculated using only numeric features
    numeric_features=X.select_dtypes(include=["number"]).columns
    corr = X[numeric_features].corrwith(y).abs()
    corr_sorted = corr.sort_values(ascending = False)
    best_features = corr_sorted.head(top_n).index.to_list()
    # print("Top features for PCA based on correlation with target:")
    # print(corr_sorted.head(top_n))
    
    return best_features

# Apply PCA
def apply_pca(X, features, standardize=True):
    X = X.loc[:, features]
    if standardize:
        X = (X-X.mean(axis=0))/X.std(axis=0)
    pca = PCA()
    X_pca = pca.fit_transform(X)
    component_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
    X_pca = pd.DataFrame(X_pca, columns = component_names)
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=component_names,
        index=X.columns
    )
    return pca, X_pca, loadings

# We can add pca components directly as features
def pca_components(df, features):
    # X = df.loc[:, features]
    _, X_pca, _ = apply_pca(df, features)
    return X_pca

# We can manually construct PCA_inspired features
def pca_inspired(df):
    X=pd.DataFrame()
    X['TotalSize'] = df.GrLivArea + df.GarageArea + df.TotalBsmtSF + df.FirstFlrSF
    X['GarageVsLiving'] = df.GarageArea - df.TotalBsmtSF - df.FirstFlrSF
    X['BasementVsFirst'] = df.TotalBsmtSF/(df.FirstFlrSF+1) # Add 1 incase there's a zero value in the denominator
    X['GarageVsCars'] = df.GarageArea/(df.GarageCars+1)
    return X

def create_final_features(df, y_train, include_cluster=True, include_pca_inspired=True, include_pca_components=True, top_pca_features=5, n_clusters=20):
    X_list = []

    # Basic mathematical transformations
    X_list.append(mathematical_transforms(df))

    # # Interaction features
    X_list.append(interactions(df))

    # # Counts of porch types
    X_list.append(counts(df))

    # Break down MSSubClass
    # X_list.append(break_down(df))

    # Group transformations
    # X_list.append(group_transform(df))

    # KMeans clustering features
    # if include_cluster:
        # X_list.append(cluster_labels(df, cluster_features, n_clusters=n_clusters))
        # X_list.append(cluster_distance(df, cluster_features, n_clusters=n_clusters))

    # # PCA-inspired features
    if include_pca_inspired:
        X_list.append(pca_inspired(df))

    # Actual PCA components based on top correlated features
    # if include_pca_components:
    #     best_features = select_features_to_apply_pca(df, y_train)

    #     X_list.append(pca_components(df, best_features))

    # Concatenate all features
    X_final = pd.concat(X_list, axis=1)
    
    # # Fill missing values if any (just in case)
    X_final = X_final.fillna(0.0)

    return X_final

if __name__=="__main__":
    df_train, df_test = load_data()
    target_col = "SalePrice"
    y_train = df_train[target_col]

    # ---------------------------------------------------------
    # STEP 1: Baseline 
    # ---------------------------------------------------------
    X_train_raw = df_train.drop(columns=target_col)
    # Use errors='ignore' because the test set doesn't have SalePrice
    X_test_raw = df_test.drop(columns=target_col, errors='ignore') 

    # We use .copy() so we don't permanently alter the raw data
    X_train_baseline = encode_and_impute(X_train_raw.copy()).astype(float)
    baseline_score = score_dataset(X_train_baseline, y_train)
    print(f"Baseline score: {baseline_score:.5f} RMSLE")

    # ---------------------------------------------------------
    # STEP 2: Mutual Information (Drop Uninformative)
    # ---------------------------------------------------------
    X_train_mi = X_train_raw.copy()
    X_test_mi = X_test_raw.copy()

    mi_scores = make_mi_scores(X_train_mi, y_train)
    
    # CRITICAL: Apply the exact same drops to BOTH train and test
    X_train_mi = drop_uninformative(X_train_mi, mi_scores)
    X_test_mi = drop_uninformative(X_test_mi, mi_scores)

    X_train_mi_encoded = encode_and_impute(X_train_mi.copy()).astype(float)
    score_with_mi = score_dataset(X_train_mi_encoded, y_train)
    print(f"Score with MI applied: {score_with_mi:.5f} RMSLE")

    # ---------------------------------------------------------
    # STEP 3: Feature Engineering
    # ---------------------------------------------------------
    X_train_fe = X_train_mi.copy()
    X_test_fe = X_test_mi.copy()

    # Create and join engineered features for Train
    final_features_train = create_final_features(X_train_fe, y_train)
    X_train_fe = X_train_fe.join(final_features_train)

    # Create and join engineered features for Test 
    # (Assuming y_train is optional/omitted for test sets in your function)
    final_features_test = create_final_features(X_test_fe, y_train) 
    X_test_fe = X_test_fe.join(final_features_test)

    X_train_fe_encoded = encode_and_impute(X_train_fe.copy()).astype(float)
    score_with_fe = score_dataset(X_train_fe_encoded, y_train)
    print(f"Score after feature engineering: {score_with_fe:.5f} RMSLE")

    # ---------------------------------------------------------
    # STEP 4: Final Preparation & Submission
    # ---------------------------------------------------------
    # THE CONCAT TRICK: To prevent column mismatches during One-Hot Encoding, 
    # we temporarily combine train and test, encode them together, then split them.
    X_combined = pd.concat([X_train_fe, X_test_fe])
    X_combined_encoded = encode_and_impute(X_combined).astype(float)

    # Split them back apart using their original indices
    X_train_final = X_combined_encoded.loc[X_train_fe.index]
    X_test_final = X_combined_encoded.loc[X_test_fe.index]

    # Train final model
    xgb = XGBRegressor(
        n_estimators = 100,
        learning_rate = 0.1,
        max_depth = 5,
        random_state = 42
    )
    
    # Fit on training data
    log_y = np.log(y_train + 1e-9)
    xgb.fit(X_train_final, log_y)
    
    # Predict on test data
    predictions_log = xgb.predict(X_test_final)
    # Reverse the log transformation
    predictions = np.exp(predictions_log) - 1e-9 
    
    # Save submission
    output = pd.DataFrame({'Id': X_test_final.index, 'SalePrice': predictions})
    output.to_csv('out/house_price_prediction_results.csv', index=False)
    print("Your submission was successfully saved! Ready for the leaderboard.")

