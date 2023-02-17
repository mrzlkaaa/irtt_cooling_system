import pytest



from preprocessing import path, CsvRefactorer, DataPreprocess

IDs = [481, 309, 317, 319]

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


# @pytest.fixture
# def DP():
#     return DataPreprocess()

# def test_get_transformer(DP):
#     numerical = DP.get_transformer("numerical")
#     assert numerical.__name__ == "numerical_encode"