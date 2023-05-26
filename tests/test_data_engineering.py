from app.data_engineering import FeatureEngineering
import pytest
import pandas as pd
import os

from preprocessing import CsvRefactorer
from data_engineering import PeriodicDataPreprocess

path_to = os.path.join(
    os.path.dirname(__file__),
    "to_preprocessing_tests.csv"
)


    
@pytest.fixture
def PDP():
    
    df = pd.read_csv(path_to, index_col="Timestamp")
    period = CsvRefactorer.select_time_period(
        df,
        [
            ("2022-10-17","2022-10-21"), 
            ("2022-11-08","2022-11-11"), ("2022-11-15","2022-11-18")
        ]
    )
    
    return PeriodicDataPreprocess(period)

def test_conditional_rows_drop(PDP):
    pumps = ["p21", "p22", "p23", "p24"]
    PDP.conditional_rows_drop(pumps, "eq", 0.0,)
    
    with pytest.raises(KeyError):
        PDP.period["2022-11-15","2022-11-18"].loc["2022-11-15 09:30:00", pumps]

    # assert 0
    
def test_pumps_mapping(PDP):
    PDP.pumps_mapping()

    with pytest.raises(KeyError):
        PDP.period["2022-11-15","2022-11-18"].loc[:, "pumps2"]

def test_filter_by_deviation(PDP):
    print(PDP.period["2022-11-15 2022-11-18"])
    PDP.filter_by_deviation("Q2", 0.1)
    print(PDP.period["2022-11-15 2022-11-18"])

    assert 0

def test_to_dataframe(PDP):
    
    df = PDP.to_dataframe()
    assert type(PDP.period) != type(df)
    


@pytest.fixture
def FE():
    
    df = pd.read_csv(path_to, index_col="Timestamp")
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
    FE.make_time_onpower_feature(time_periods=periods)
    assert 0


