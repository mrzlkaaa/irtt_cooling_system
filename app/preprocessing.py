from __future__ import annotations
from calendar import c
from operator import index
from typing import Dict, List, Tuple, Union, Set
from collections import defaultdict
import pandas as pd
import numpy as np
import os

import asyncio

path = os.path.join(os.path.split(os.path.dirname(__file__))[0], "jupyter", "041022_to_231222.csv")

#! rename to DfRefactorer
class CsvRefactorer:
    
    '''
   #* Class purpose is to refactor input csv with given columns
   #* <ID, Time, Value, Quality> to more readable style with columns 
   #* < Time ID1 val ID2 val ... IDn val >
   #* The recommended order of methods use:
   #* 1. select_by_ids:
   #*   returns List[pd.core.frame.DataFrame]
   #*   where number of dfs is equal of number of a given ids 
   #* 2. min_frac_groupby (optional):
   #*   group the data by a given time frequancy
   #* 3. concat_dfs:
   #*   concatenate dfs from previuos steps
   #*   by columns (axis=1) so each ID becomes a column
   #*   and each row assotiates with a particular time
   #* Attributes
   #* ----------
   #*
   #* Methods
   #* ----------
   #*
   '''
    IDS_MAP = {
            299: "T1aHE",
            309: "P2",
            315: "T1bHE",
            317: "T2bHE",
            319: "T2aHE",
            321: "Treactor",
            325: "T2aHE1",
            327: "Tair",
            381: "CTF1",
            395: "CTF2",
            396: "CTF3",
            460: "T2aHE2",
            461: "T2aHE3",
            462: "T2aHE4",
            463: "T2aHE5",
            481: "Q2",
            406: "p21",
            407: "p22",
            408: "p23",
            409: "p24",
        }

    def __init__(self, 
        df: pd.core.frame.DataFrame, 
        quickclean: bool = True, 
        index_range: Union[
            Tuple[str, str],
            Tuple[np.datetime64,np.datetime64], 
            None
        ] = None
    ) -> None:
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

    #* immutable
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

    def ids_mapping(self):

        return self.IDS_MAP

    def select_by_ids(
        self, 
        ids: List[int] = None
    ) -> List[pd.core.frame.DataFrame]:
        '''
        #* query and select rows with a given id
        #* creates a list of dfs where each dataframe 
        #* consists of only 1 queried id
        #* Parameters
        #* ----------
        #* ids: List[int]
        #*  array of ids
        #* Returns
        #* ----------
        #* array of dataframes - each id has its own dataframe
        '''
        if ids is None:
            ids = self.ids
        
        if not isinstance(ids, list):
            ids = list(ids)
    
        series: List[pd.core.frame.DataFrame] = []

        for id in ids: 
            series.append(self.df[self.df["ID"] == id])
        return series


    def min_frac_groupby(
        self,
        frac: Union[str, int, float] = "5", 
        *dfs: Tuple[pd.core.frame.DataFrame]
    ) -> List[pd.core.frame.DataFrame]:
        """
        *groupby given dataframes by given frequancy in unit of minutes
        *if the output length do not equal for all dfs raise ValueError
        """
        if isinstance(frac, int) or isinstance(frac, float):
            frac = str(frac) 
        series: List[pd.core.frame.DataFrame] = []
        for df in dfs:
            series.append(df.groupby(pd.Grouper(freq=f"{frac}min")).mean())
        # self.series_len_check(series)
        return series

    def series_len_check(self, series: List[pd.core.frame.DataFrame]) -> None:
        """
        *accepts series where each df grouped by frequancy
        *function creates the array of length where each length corresponds with df
        *after converts array to a set that cause to drop duplitates 
        *if the length of set is greater 1 raise ValueError (dfs are not equal)
        """
        lens: List[pd.core.frame.DataFrame] = []
        for i in series:
            lens.append(len(i))
        lens_set: Set[pd.core.frame.DataFrame] = set(lens)
        if len(lens_set) > 1:
            raise ValueError("selected ids have different lenghts")

    def concat_dfs(
        self, 
        dfs: list, 
        cols_names: List[str] | None = None
    ) -> pd.core.frame.DataFrame:
        '''
        #* Method description
        #* Parameters
        #* ----------
        #*
        #* Raises
        #* ----------
        #*
        #* Returns
        #* ----------
        #*
        '''
        
        if cols_names is None:
            #* retrieve and use default cols_names
            cols_names = [int(i.iloc[0]["ID"]) for i in dfs]
        
        #* drops ID col from on df in a list
        dfs = [i.drop(["ID"], axis=1) for i in dfs]
    
        return pd.concat(dfs, axis=1).set_axis(cols_names, axis=1)
    
    #* accepts dfs for futher creation of new df
    def create_df_from_dfs(
        self, 
        dfs: list, 
        param: str = "ID"
    ) -> pd.core.frame.DataFrame:
        """
        * Can be applied only on dfs with the same length
        * Create DataFrame  with following structure < Time ID1 val ID2 val ... IDn val >
        * param is parameter value which will be used as column in output DataFrame 
        * param is ID by default here
        """
        cols_value = dict()
        #* must be taken index of the longest df
        dfs.sort(key=lambda x: len(x))
        index = dfs[0].index 
        for df in dfs:
            cols_value[int(df[param][0])] = df["Value"].to_numpy()
        return pd.DataFrame(data=cols_value, index=index)

    @classmethod
    def read_csv(
        cls, path: 
        str, quickclean: 
        bool = True, 
        index_range: Union[
            Tuple[str, str],
            Tuple[np.datetime64,np.datetime64], 
            None
        ] = None,
        nrows = None,
    ) -> CsvRefactorer:
        
        if nrows:
            df: pd.core.frame.DataFrame = pd.read_csv(path, nrows=nrows)
        else:
            df: pd.core.frame.DataFrame = pd.read_csv(path)
        
        return cls(df, quickclean=quickclean, index_range=index_range)

    @staticmethod
    def drop_if_below(dfs: List[pd.core.frame.DataFrame],
            col_name: Union[str, int, float], 
            value: Union[int, float]) -> List[pd.core.frame.DataFrame]:
        """
        *helper to drop rows that do not satisfy basic logical operation > (more)
        """
        series: List[pd.core.frame.DataFrame] = []
        for df in dfs:
            series.append(df[df[col_name] > value])
        return series

    @staticmethod
    def select_time_period(
        df: pd.core.frame.DataFrame,
        periods: List[Tuple[str]],
        dropna: bool=False
    ) -> dict:
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
            
            selected_periods[f"{s} {f}"] = df.loc[s:f].fillna(0.0)
            
            if dropna:
                selected_periods[f"{s} {f}"] = selected_periods[f"{s} {f}"].dropna()

        return selected_periods

    #todo rename to < dfs_range_filter >
    @staticmethod
    def dfs_formatting(dfs: List[pd.core.frame.DataFrame], 
            keys: List[Union[str, int, float]],
            filter_range: List[Union[str, int, float]]) ->\
            Dict[str, pd.core.frame.DataFrame]:
        
        """
        * creates dict where key - parameter or name of df,
        * value - formatted df
        * value formats by range (index range actually)
        """
        store: Dict[Union[str, int, float], 
            pd.core.frame.DataFrame] = dict()
        for n, df in enumerate(dfs):
            store[keys[n]] = df.sort_index().loc[filter_range[0] : filter_range[-1]]
        return store
        
    @staticmethod
    def merge_dfs_by_tp(
        d: List[Dict[str, pd.core.frame.DataFrame]],
        time_periods: List[str],
        index: Union[int, None] = None
    )-> Dict[str, List[pd.core.frame.DataFrame]]:
        """
        * accepts output of <select_time_period> static method
        * d is list of dictionaries with keys (time period) and 
        * values (dataframe for this period)
        * where each list is dictionary (as many dataframes were
        * as many dictionaries are) splitted on period of time
        * given list can be refactored (merged) by period of times only
        * so the number of dictionaries will be equal to the number of time periods
        * gives a lot of profit when there are lots of dfs in dictionary with different length
        * < index > argument is index of dataframe upon which index to filter other dfs
        """
        tp_merged = defaultdict(list)
        #* iterates over the list of dictionaries
        for dct in d:
            #* iterates over the keys dictionary
            for tp in time_periods:
                #* append only dict with tp
                if index is not None:
                    tp_merged[tp].append(dct[tp].filter(items=d[index][tp].index, axis=0)) 
                else:
                    tp_merged[tp].append(dct[tp]) 
        return tp_merged

    @staticmethod
    def create_df_for_tp(
        d: Dict[str, List[pd.core.frame.DataFrame]], 
        time_periods: List[str]
    ) -> Dict[str, pd.core.frame.DataFrame]:
        """
        * method aim is to create df by concatenation dfs for given time period
        * concat dfs along indexes column
        * fills missing data by zeros
        """
        d_by_tp = dict()
        for tp in time_periods:
            d_by_tp[tp] = pd.concat([*d[tp]], axis=1).fillna(0.0)
        return d_by_tp


    @staticmethod
    def export_df(df: pd.core.frame.DataFrame) -> None:
        df.to_csv("123.csv", index=True) #* given name is a test


class ReactorControlSystem:
    '''
   #* The class is created to preprocess / parse
   #* .txt daily archive files outputed from MIRTEST software
   #* The general idea is to retrieve columns and merge files
   #* to a single dataframe with common Timestamp index
   #* Attributes
   #* ----------
   #*
   #* Methods
   #* ----------
   #*
   '''
   #* build a path to jupyter folder
    GENERAL_PATH = os.path.join(
        os.path.split(
            os.path.dirname(__file__)
        )[0], 
        "jupyter"
    )
    
    def __init__(
        self,
        files: list,
        columns: list | None = None,
        path: str = ""
    ):
        self.path = os.path.join(self.GENERAL_PATH, path)  #* build path to files
        self.files = files  #* order is important
        self.columns = columns
        return

    async def retrieve_from_file(self, file):
        '''
        #* Read the file, split columns and convert
        #* list to dataframe
        #* Parameters
        #* ----------
        #*
        #* Raises
        #* ----------
        #*
        #* Returns
        #* ----------
        #*
        '''

        with open(
            os.path.join(self.path, file + ".txt"), 
            "r",
            errors="ignore"
        ) as f:
            data = f.readlines()[14:]
        
        df  = pd.DataFrame(
            list(map(lambda x: x.split(), data)),
            columns=self.columns
        )
        return df

    async def get_raw_data(self):
        '''
        #! add code to work with 2d array -> retirieving 2d structure 
        #* Top async method to accelerate data
        #* retrieving from files
        #* The asyncio.gather is used to grab all independet instances
        #* Returns
        #* ----------
        #* dfs: List[df]
        '''
        dfs = await asyncio.gather(
            *[self.retrieve_from_file(i) for i in self.files]
        )
        return dfs


    def merge_days(
        self,
        dfs: list,
        st_datatime: str,
        datetime_col: str = "Timestamp",
        update_index: bool = True
    ):
        for i in range(len(dfs)):
            dfs[i].loc[:, datetime_col] = dfs[i].loc[:, datetime_col]\
                .apply(
                    lambda x: pd.to_timedelta(x) + pd.to_datetime(st_datatime) if i == 0 
                    else pd.to_timedelta(x) + dfs[i-1].iloc[-1][datetime_col]
                )
        df = pd.concat([*dfs], axis=0)

        if update_index:
            df = df.set_index(datetime_col)
            df = df.astype("float64")
        
        return df


    