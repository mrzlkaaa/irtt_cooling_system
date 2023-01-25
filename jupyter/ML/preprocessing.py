from __future__ import annotations
from typing import Dict, List, Tuple, Union, Set
import pandas as pd
import numpy as np
import os

path = os.path.join(os.path.split(os.path.dirname(__file__))[0], "041022_to_231222.csv")

class CsvRefactorer:
    """ Class purpose is to refactor input csv with given columns
    <ID, Time, Value, Quality> to more readable style with columns 
    < Time ID1 val ID2 val ... IDn val >"""
    def __init__(self, df: pd.core.frame.DataFrame, quickclean: bool = True) -> None:
        self.df:  pd.core.frame.DataFrame = df
        if quickclean:
            self.df = self.quick_clean()
        self.ids: List[int] = self.df["ID"].unique()

    @classmethod
    def read_csv(cls, path: str) -> CsvRefactorer:
        df: pd.core.frame.DataFrame = pd.read_csv(path)
        return cls(df)

    @staticmethod
    def export_df(df: pd.core.frame.DataFrame):
        df.to_csv("123.csv", index=True) #* given name is a test

    @staticmethod
    def select_time_period(df, periods):
        selected_periods: dict = dict()
        for period in periods:
            s,f = period
            selected_periods[f"{s} {f}"] = df.loc[s:f].dropna()
        return selected_periods

    #* drops nan and all quality's except 1
    #* drops rows with value == 0.0
    #* converts obj to datetime format
    #* sets timestamp as an index
    def quick_clean(self) -> pd.core.frame.DataFrame:
        df: pd.core.frame.DataFrame = self.df.dropna()
        df = df[(df["Quality"] == 1) & (df["Value"] != 0.0)] \
            .reset_index().drop(["index", "Quality"], axis=1)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], format='%Y-%m-%dT%H:%M')
        df = df.set_index('Timestamp')
        return df

    def select_by_ids(self, ids: List[int]) -> List[pd.core.frame.DataFrame]:
        if not isinstance(ids, list):
            ids = list(ids)
        if len(ids) == 0:
            ids = self.ids
        series: List[pd.core.frame.DataFrame] = []
        for id in ids: 
            series.append(self.df[self.df["ID"] == id])
        return series

    def min_frac_groupby(self, frac: Union[str, int, float] = "5", *dfs: pd.core.frame.DataFrame) -> List[pd.core.frame.DataFrame]:
        if isinstance(frac, int) or isinstance(frac, float):
            frac = str(frac) 
        series: List[pd.core.frame.DataFrame] = []
        for df in dfs:
            series.append(df.groupby(pd.Grouper(freq=f"{frac}min")).mean())
        self.series_len_check(series)
        return series

    def series_len_check(self, series: List[pd.core.frame.DataFrame]) -> None:
        lens: List[pd.core.frame.DataFrame] = []
        for i in series:
            lens.append(len(i))
        lens_set: Set[pd.core.frame.DataFrame] = set(lens)
        if len(lens_set) > 1:
            raise ValueError("selected ids have different lenghts")

    #* accepts dfs for futher creation of new df
    def create_df_from_dfs(self, param, dfs) -> pd.core.frame.DataFrame:
        cols_value = dict()
        index = dfs[0].index
        for df in dfs:
            cols_value[int(df[param][0])] = df["Value"].to_numpy()
        return pd.DataFrame(data=cols_value, index=index)

