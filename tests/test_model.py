import pytest
import pandas as pd
import numpy as np
import os
from app.model import *

from sklearn.preprocessing import StandardScaler

# df_test = pd.read_excel(
#     os.path.join(
#         os.path.split(
#             os.path.dirname(__file__)
#         )[0],
#         "jupyter",
#         "rdy_for_stats_121020_210423.xlsx"    
#     ),
#     nrows=1000)

df_test = pd.read_excel(
    os.path.join(
        os.path.split(
            os.path.dirname(__file__)
        )[0],
        "tests",
        "tests_model.xlsx"    
    ),
    index_col=0
)

X_arr = [
    [-2.320476  , -1.43139267, -0.4272752 , 4.5951246 ,
        1.04236079, -0.49017386],
    [-1.41798195, -0.86394491, -0.60568636,  0.13741927,
    -0.56311249, -0.49229722],
    [-0.85422427, -0.61134112,  0.85148245,  0.36942693,
        0.24859365,  0.06001498],
    [-0.56868771, -1.0387775 ,  3.04011026, -0.75370127,
    -0.10634723, -0.02137261],
    [-1.02791993, -1.45913286,  2.11247639, -0.63260712,
        0.15057512,  0.10392003],
    [-1.44554175, -1.48587308,  0.16558634, -0.00912788,
        0.70118702,  0.36607095]
]

y_arr = [-2.320476  , -1.43139267, -0.4272752 , 4.5951246 ,
        1.04236079, -0.49017386]

@pytest.fixture
def sso():
    X = df_test.drop("dt1", axis=1)
    y = df_test.loc[:, "dt1"]
    
    # return SingleStepOutput(X=X, y=y)
    return SingleStepOutput(X=X_arr, y=y_arr)

def test_default_split(sso):
    res = sso.default_split(indices=False)
    print(res)
    print(sso.test_X)
    assert 0

def test_X(sso):
    # print(sso.cv)
    print(sso.X)
    assert 0

def test_sso_forecast(sso):
    sso.default_split(indices=True)
    print(sso.test_X)
    res = sso.forecast("GBR", sso.test_X)
    print(res, sso.test_y)
    assert 0


@pytest.fixture
def preprocessing():
    X = df_test
    print(X)
    return Preprocessing(X)

def test_ct(preprocessing):
    res = preprocessing.column_transformer(
        transformers = [
            ("numerical", StandardScaler(), preprocessing.X.columns)
        ]
    )    
    print(res)
    assert 0

def test_add_shifts(preprocessing):
    preprocessing.add_shifts(
        ["T2bHE"], 
        12,
        reset_index=True
    )
    print(preprocessing.X)
    assert 0