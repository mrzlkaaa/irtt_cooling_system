from time import time
import pytest
from analytics import WaterFlowRates
import numpy as np
from .test_preprocessing import Refactorer, IDs

time_periods = [("2022-10-17", "2022-10-21"), ("2022-11-08","2022-11-11"), ("2022-11-15","2022-11-18"),
                ("2022-11-22","2022-11-25"), ("2022-11-29","2022-12-02"), ("2022-12-05","2022-12-09"), 
                ("2022-12-12","2022-12-16"), ("2022-12-19","2022-12-23")]

@pytest.fixture
def wfr():
    return WaterFlowRates(10)

@pytest.fixture
def df(Refactorer):
    series = Refactorer.select_by_ids(IDs)
    frac_series = Refactorer.min_frac_groupby("5", *series)
    return Refactorer.create_df_from_dfs("ID", frac_series)

@pytest.fixture
def df(Refactorer):
    series = Refactorer.select_by_ids(IDs)
    frac_series = Refactorer.min_frac_groupby("5", *series)
    df = Refactorer.create_df_from_dfs("ID", frac_series)
    df_periods = Refactorer.select_time_period(df, time_periods)
    return df_periods
    
@pytest.fixture
def df_filtered(wfr, df):
    df_fp = df[list(df.keys())[0]]
    df_t, df_f = wfr.df_md_filter(df_fp, 481)
    return df_t

def test_df_md_filter(wfr, df):
    df_t, df_f = wfr.df_md_filter(df, 481)
    print(df_t)
    assert 0

def test_lin_regression_fit(wfr, df_filtered):
    y = df_filtered[481].to_numpy().reshape(-1,1)
    X = np.arange(0, len(df_filtered[481]), 1).reshape(-1,1)
    lr = wfr.lin_regression_fit(X, y)
    print(lr.score(X, y))
    assert 0
    