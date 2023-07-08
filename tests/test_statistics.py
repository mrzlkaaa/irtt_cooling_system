import pytest
import os
import pandas as pd
import numpy as np
from app.statistics import *

path_to = os.path.join(
    os.path.dirname(__file__),
    "to_stats_tests.xlsx"
)

@pytest.fixture
def stats():
    df = pd.read_excel(path_to, index_col="Timestamp")
    return Statistics(df)

def test_get_iqrs(stats):
    iqrs = stats.iqrs
    print(iqrs)
    assert iqrs.get("CTF1") != None

def test_FD_rule(stats):
    bins = stats.FD_rule("Q2")
    print(bins)
    assert 0

@pytest.fixture
def NP():
    rng = np.random.default_rng()
    x = rng.random(size=35)
    return NonParametric(x)

def test_run_tests_np(NP):
    NP.run_tests()
    assert 0

@pytest.fixture
def TS():
    rng = np.random.default_rng()
    x = rng.random(size=35)
    return TimeSeries(x)

def test_adf(TS):
    TS.adf()
    assert 0

def test_run_tests_ts(TS):
    TS.run_tests()
    assert 0