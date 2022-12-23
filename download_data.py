import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle
import pandas as pd

# just define a function to load data with selected columns
def download_ames_housing():
    df = fetch_openml(name="house_prices", as_frame=True)
    X = df.data
    y = df.target

    features = [
        "YrSold",
        "HeatingQC",
        "Street",
        "YearRemodAdd",
        "Heating",
        "MasVnrType",
        "BsmtUnfSF",
        "Foundation",
        "MasVnrArea",
        "MSSubClass",
        "ExterQual",
        "Condition2",
        "GarageCars",
        "GarageType",
        "OverallQual",
        "TotalBsmtSF",
        "BsmtFinSF1",
        "HouseStyle",
        "MiscFeature",
        "MoSold",
    ]

    X = X[features]
    X, y = shuffle(X, y, random_state=0)

    X = X.iloc[:600, 0:5]
    y = np.log(y[:600])
    ames = pd.concat([X, y], axis=1)
    ames.to_csv("./data/ames.csv", index=False)
    print("datasets saved")
    

if __name__=="__main__":
    download_ames_housing()