import pandas as pd
import numpy as np
import os

path = os.path.join(os.path.split(os.path.dirname(__file__))[0], "041022_to_231222.csv")

class CsvRefactorer:
    def __init__(self, csv, quickclean=True):
        self.csv = csv
        if quickclean:
            self.csv = self.quick_clean()
        self.ids = self.csv["ID"].unique()

    @classmethod
    def read_csv(cls, path):
        csv = pd.read_csv(path)
        return cls(csv)

    @staticmethod
    def export_df(df):
        df.to_csv("123.csv", index=True)

    #* drops nan and all quality's except 1
    #* drops rows with value == 0.0
    #* converts obj to datetime format
    #* sets timestamp as an index
    def quick_clean(self):
        df = self.csv.dropna()
        df = df[(df["Quality"] == 1) & (df["Value"] != 0.0)]
        df = df.reset_index().drop(["index", "Quality"], axis=1)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], format='%Y-%m-%dT%H:%M')
        df = df.set_index('Timestamp')
        return df

    def select_by_ids(self, *ids):
        if len(ids) == 0:
            ids = self.ids
        series = []
        for id in ids: 
            series.append(self.csv[self.csv["ID"] == id])
        return series

    def min_frac_groupby(self, frac="5", *dfs):
        series = []
        for df in dfs:
            series.append(df.groupby(pd.Grouper(freq=f"{frac}min")).mean())
        self.series_len_check(series)
        return series

    def series_len_check(self, series):
        lens = []
        for i in series:
            lens.append(len(i))
            print(len(i))
        lens = set(lens)
        if len(lens) > 1:
            raise ValueError("selected ids have different lenghts")

    def select_time_period(self, df, *periods):
        selected_periods = dict()
        for period in periods:
            s,f = period
            selected_periods[f"{s} {f}"] = df.loc[s:f].dropna()
        return selected_periods


    #* accepts dfs for futher creation of new df
    def create_df_from_dfs(self, param, dfs):
        cols_value = dict()
        cols_name = []
        index = dfs[0].index
        for df in dfs:
            cols_value[int(df[param][0])] = df["Value"].to_numpy()
        dataframe = pd.DataFrame(data=cols_value, index=index)
        print(dataframe)
        return dataframe

