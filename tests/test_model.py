import pytest
import pandas as pd
import numpy as np
import os
from app.model import *

df_test = pd.read_excel(
    os.path.join(
        os.path.split(
            os.path.dirname(__file__)
        )[0],
        "jupyter",
        "rdy_for_stats_121020_210423.xlsx"    
    ),
    nrows=1000)

@pytest.fixture
def sso():
    X = df_test.drop("dt1", axis=1)
    y = df_test.loc[:, "dt1"]
    
    return SingleStepOutput(X=X, y=y)

def test_default_split(sso):
    res = sso.default_split(indices=False)
    print(res)
    assert 0

def test_X(sso):
    # print(sso.cv)
    print(sso.X)
    assert 0