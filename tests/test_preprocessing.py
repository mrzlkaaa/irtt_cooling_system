import pytest
import pandas as pd
import os
import asyncio

from preprocessing import (
    path, 
    CsvRefactorer, 
    # DataPreprocess, 
    # PeriodicDataPreprocess,
    ReactorControlSystem
)

IDs = [481, 309, 317, 319]

df_test = pd.read_csv(
    os.path.join(
        os.path.dirname(__file__), "to_test.csv"
    ),
    index_col=[0])

@pytest.fixture
def Refactorer():
    # print(path)
    return CsvRefactorer.read_csv(path)

@pytest.fixture
def Refactorer_ran_ind():
    # print(path)
    return CsvRefactorer.read_csv(path, quickclean=True, index_range = ("20221017", "20221021"))

def test_read_df(Refactorer):
    print(Refactorer.df)
    assert 0

def test_read_df_ran_ind(Refactorer_ran_ind):
    print(Refactorer_ran_ind.df)
    assert 0

def test_select_by_ids(Refactorer):
    series = Refactorer.select_by_ids(IDs)
    print(series)
    assert 0

def test_min_frac_groupby(Refactorer):
    series = Refactorer.select_by_ids(IDs)
    frac_series = Refactorer.min_frac_groupby(5, *series)
    print(frac_series)
    assert 0

def test_select_time_period(Refactorer):
    series = Refactorer.select_by_ids(IDs)
    frac_series = Refactorer.min_frac_groupby("5", *series)
    ids_period = dict()
    for i in frac_series:
        ids_period[i["ID"][0]] = Refactorer.select_time_period(i, ("2022-10-17", "2022-10-21"))
    print(ids_period)
    assert 0

def test_create_df_from_dfs(Refactorer):
    series = Refactorer.select_by_ids(IDs)
    frac_series = Refactorer.min_frac_groupby("5", *series)
    Refactorer.create_df_from_dfs("ID", frac_series)
    assert 0

def test_export_df(Refactorer):
    series = Refactorer.select_by_ids(IDs)
    frac_series = Refactorer.min_frac_groupby("5", *series)
    df = Refactorer.create_df_from_dfs("ID", frac_series)
    Refactorer.export_df(df)

def test_drop_if_below(Refactorer):
    print(Refactorer.drop_if_below([Refactorer.df], "ID", 30.0))
    assert 0

def test_concat_dfs(Refactorer):
    df = Refactorer.concat_dfs([df_test])
    print(df)
    assert 0


# path_to_PDP = os.path.join(
#     os.path.split(
#         os.path.dirname(__file__)
#     )[0],
#     "jupyter",
#     "P2_second_circuit_data_050922_to_210423.csv"
# )


@pytest.fixture
def RCS():
    return ReactorControlSystem(
        path="control_operation_system_params",
        files=["20221017", "20221018", "20221019", "20221020", "20221021"]
    )

@pytest.mark.asyncio
async def test_get_raw_data(RCS):
    await RCS.get_raw_data()
    assert 0

@pytest.mark.asyncio
async def test_merge_days(RCS):
    raw_dfs = await RCS.get_raw_data()
    df = RCS.merge_days(raw_dfs, "2022-10-17 10:05:30", 0)
    print(df)
    assert 0
