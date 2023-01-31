from __future__ import annotations
from operator import index
from typing import Dict, List, Tuple, Union, Set
import pandas as pd
import numpy as np
import os

path = os.path.join(os.path.split(os.path.dirname(__file__))[0], "jupyter", "041022_to_231222.csv")

class CsvRefactorer:
    """ 
    * Class purpose is to refactor input csv with given columns
    * <ID, Time, Value, Quality> to more readable style with columns 
    * < Time ID1 val ID2 val ... IDn val >
    """
    def __init__(self, df: pd.core.frame.DataFrame, quickclean: bool = True, 
            index_range: Union[Tuple[str, str],
                         Tuple[np.datetime64,np.datetime64], 
                         None] = None) -> None:
        """
        *index_range can be applied only if quick_clean is set to True
        """
        self._df:  pd.core.frame.DataFrame = df
        if quickclean:
            self._df = self.quick_clean()
            if index_range is not None:
                self._df:  pd.core.frame.DataFrame = self._df.sort_index()\
                    .loc[index_range[0]:index_range[-1], :]
        self.ids: List[int] = self._df["ID"].unique()

    @property
    def df(self) -> pd.core.frame.DataFrame:
        return self._df

    @df.setter
    def df(self, val) -> None:
        self._df = val

    def quick_clean(self) -> pd.core.frame.DataFrame:
        """
        *drops nan and all quality's except 1
        *drops rows with value == 0.0
        *converts obj to datetime format
        *sets timestamp as an index
        """
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
    def create_df_from_dfs(self, param: str, dfs: list) -> pd.core.frame.DataFrame:
        """
        * Create DataFrame  with following structure < Time ID1 val ID2 val ... IDn val >
        * param is parameter value which will be used as column in output DataFrame 
        * param is ID by default here
        """
        cols_value = dict()
        index = dfs[0].index
        for df in dfs:
            cols_value[int(df[param][0])] = df["Value"].to_numpy()
        return pd.DataFrame(data=cols_value, index=index)

    @classmethod
    def read_csv(cls, path: str, quickclean: bool = True, 
            index_range: Union[Tuple[str, str],
                         Tuple[np.datetime64,np.datetime64], 
                         None] = None) -> CsvRefactorer:
        df: pd.core.frame.DataFrame = pd.read_csv(path)
        return cls(df, quickclean=quickclean, index_range=index_range)

    @staticmethod
    def drop_if_below(dfs: List[pd.core.frame.DataFrame],
            col_name: Union[str, int, float], 
            value: Union[int, float]) -> pd.core.frame.DataFrame:
        """
        helper to drop rows that do not satisfy basic logical operation >
        """
        series: List[pd.core.frame.DataFrame] = []
        for df in dfs:
            series.append(df[df[col_name] > value])
        return series

    @staticmethod
    def select_time_period(df, periods):
        """
        * takes  DataFrame and TimeStamps 
        * of certain period (started and finished)
        * iterate over given periods (array) and select rows via .loc
        * creates dictionary where the key is 
        * time period presented as <started finished> (where blank is delimeter)
        """
        selected_periods: dict = dict()
        for period in periods:
            s,f = period
            selected_periods[f"{s} {f}"] = df.loc[s:f].dropna()
        return selected_periods

    @staticmethod
    def export_df(df: pd.core.frame.DataFrame):
        df.to_csv("123.csv", index=True) #* given name is a test
