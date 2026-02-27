# Feature Engineering - Creating new features
import pandas as pd

df = pd.read_csv("../Datasets/FE Course Data/ames.csv")

# 1. Create new features through mathematical transformations
X_1 = pd.DataFrame()
X = df.copy()
y = X.pop("SalePrice")

X_1['LivLotRatio'] =  X["GrLivArea"] / X["LotArea"]
X_1["Spaciousness"] = (X["FirstFlrSF"] + X["SecondFlrSF"]) / X["TotRmsAbvGrd"]
X_1["TotalOutsideSF"] = (
    X["WoodDeckSF"] +
    X["OpenPorchSF"] +
    X["EnclosedPorch"] +
    X["Threeseasonporch"] +
    X["ScreenPorch"]
)

# 2. Interaction with a categorial
X_2 = pd.get_dummies(df.BldgType, prefix="Bldg")
X_2 = X_2.mul(df.GrLivArea, axis=0)
# print(X_2)

# 3. Creating features by counting each values in a data column
X_3 = pd.DataFrame()

X_3["PorchTypes"] = df[[
    "WoodDeckSF",
    "OpenPorchSF",
    "EnclosedPorch",
    "Threeseasonporch",
    "ScreenPorch",
]].gt(0.0).sum(axis=1)
# print(X_3)

# 4. Breakdown a categorical feature
X_4 = pd.DataFrame()

X_4["MSClass"] = df.MSSubClass.str.split("_", n=1, expand=True)[0]
# print(X_4)

# 5. Use a grouped transform
X_5 = pd.DataFrame()
# median of GrLivArea grouped on Neighborhood
X_5["MedNhbdArea"] = df.groupby("Neighborhood")["GrLivArea"].transform("median")
print(X_5)

X_new = X.join([X_1, X_2, X_3, X_4, X_5])
# score_dataset(X_new, y)