from ast import operator
from app.data_engineering import FeatureEngineering
import pytest
import pandas as pd
import os

from preprocessing import CsvRefactorer
from data_engineering import PeriodicDataPreprocess

path_to = os.path.join(
    os.path.dirname(__file__),
    "to_stats_tests.xlsx"
)


pumps = ["p21", "p22", "p23", "p24"]

@pytest.fixture
def PDP():
    
    df = pd.read_excel(path_to, index_col="Timestamp")
    period = CsvRefactorer.select_time_period(
        df,
        [
            ("2022-10-17","2022-10-21"), ("2022-11-08","2022-11-11"), 
            ("2022-11-15","2022-11-18"), ("2022-11-29","2022-12-02")
        ]
    )
    
    return PeriodicDataPreprocess(period)


def test_conditional_rows_drop(PDP):
    

    # PDP.conditional_rows_drop(condition="eq", value=0.0)
    PDP.conditional_rows_drop(pumps, "eq", 0.0)

    with pytest.raises(KeyError):
        PDP.period["2022-11-15","2022-11-18"].loc["2022-11-15 09:30:00", pumps]

    assert 0
    
def test_pumps_mapping(PDP):
    PDP.pumps_mapping(pumps)
    assert 0

    with pytest.raises(KeyError):
        PDP.period["2022-11-15","2022-11-18"].loc[:, "pumps2"]

def test_filter_by_deviation(PDP):
    print(PDP.period["2022-11-15 2022-11-18"].shape)
    PDP.filter_by_deviation(column="Q2", bysigma=2)
    # PDP.filter_by_deviation("Q2", 0.1)
    print(PDP.period["2022-11-15 2022-11-18"].shape)

    assert 0

def test_filter_by_zscore(PDP):
    print(PDP.period["2022-11-15 2022-11-18"].shape)
    PDP.filter_by_zscore(column="Q2", pvalue=0.90)
    # PDP.filter_by_deviation("Q2", 0.1)
    print(PDP.period["2022-11-15 2022-11-18"].shape)

    assert 0

def test_to_dataframe(PDP):
    
    df = PDP.to_dataframe()
    assert type(PDP.period) == type(df)
    
def test_sma_smoothing(PDP):
    df_before = PDP.period["2022-11-15 2022-11-18"].copy()
    df_after = PDP.sma_smoothing(10)["2022-11-15 2022-11-18"].copy()
    print(df_before, df_after)
    assert df_after.shape < df_before.shape
    assert 0

row = "2022-12-01 23:40:00"

@pytest.fixture
def FE():
    
    df = pd.read_excel(path_to, index_col="Timestamp")
    return FeatureEngineering(df)

def test_columns_averaging(FE):
    
    df = FE.columns_averaging(["p21", "p22", "p23"], feature_name ="I2mean")
    print(df)
    assert 0

def test_make_time_onpower_feature(FE):
    periods = [
        ("2022-10-17","2022-10-21"), 
        ("2022-11-08","2022-11-11"), ("2022-11-15","2022-11-18")
    ]
    # periods = [( "2020-09-05", "2023-04-15")]

    df = FE.make_time_onpower_feature(start=[2000, "", ""], time_periods=periods)
    print(df)
    assert 0

def test_pumps_normalizer(FE, PDP):
    PDP.pumps_mapping(["p21", "p22", "p23", "p24"])
    
    FE.df = PDP.to_dataframe()
    # print(FE.df)
    df_before = FE.df.copy()
    df_after = FE.pumps_normalizer("Q2")
    

    assert df_after.loc[row, ["Q2"]].to_numpy() > df_before.loc[row, ["Q2"]].to_numpy()

def test_make_dts_on_HEs(FE):
    df_before = FE.df.copy()
    df_after = FE.make_dts_on_HEs(inplace=True)

    assert df_before.loc[row, ["T2aHE1"]].to_numpy() > df_after.loc[row, ["T2aHE1"]].to_numpy()

def test_conditional_rows_drop_fe(FE):
    FE.conditional_rows_drop(pumps, "eq", 0.0)

    with pytest.raises(KeyError):
        FE.df.loc["2022-11-15 09:30:00", pumps]

def test_columns_categorizing(FE):
    df = FE.columns_categorizing(
        columns = ["CTF1", "CTF2", "CTF3"], 
        value = 20,
        i = 1,
        e = 0,
        operator = "gt"
    )
    print(df["CTF1"].describe())
    assert 0

