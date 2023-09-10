from __future__ import annotations
from calendar import leapdays
from logging import critical
import re
from xml.etree.ElementInclude import include
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Tuple, Union, Set
from collections import defaultdict
from scipy import stats

# from app import *
critical_values = {
        0.90: (-1.65, 1.65),
        0.95: (-1.96, 1.96),
        0.99: (-2.58, 2.58)
    }


class DataPreprocess(ABC):
    PUMPS_MAP = {
        "p21": "1",
        "p22": "2",
        "p23": "3",
        "p24": "4",
    }

    PUMPS_COEF = {
        "123": 1.03,
        "124": 1.02,
        "234": 1
    }
    
    OPMV_MAP = {
        "Q2": 400,
        "dt1": 4.0,
        "dt2": 4.0,
        "Ipumps": 50.0
    }

    CRITICAL_VALUES = critical_values

    def __init__(self):
        return

    @abstractmethod
    def _make_default_features(self):
        raise NotImplementedError

    @abstractmethod
    def conditional_rows_drop(self):
        raise NotImplementedError

    @abstractmethod
    def filter_by_deviation(self):
        raise NotImplementedError

    def _get_binary_operator(
        self,
        obj: object,
        operator: str
    ) -> Callable:
        return getattr(type(obj), operator)

    @staticmethod
    def retrieve_datatime(
        series: pd.core.frame.Series,
        attr: str,
        byindex: bool = True,
        
    ) -> Union[pd.core.frame.Series, None]:
        #! does not work if pass DateTime as index column
        """
        * retrieves required data (day, hour and etc.) from Timestamp (datetime obj)
        * by attribute name
        * returns new series of required attributes
        * raise attribute error if there is no data given in TimeStamp obj
        """
        return series.apply(lambda x: getattr(x, attr))


class PeriodicDataPreprocess(DataPreprocess):

    def __init__(
        self, 
        period: Dict[pd.core.frame.DataFrame],
        default_features: bool = True
    ) -> None:

        super().__init__()
        self._period = self._nan_handler(period)
        self.period_items = (i for i in self.period.items())
        self._period_keys = self.period.keys()

        if default_features:
            self._make_default_features()

    def _make_default_features(self) -> None:
        for i in self.period_keys:
            self.period[i]["dt1"] = self.period[i]["T1bHE"] - self.period[i]["T1aHE"]
            self.period[i]["dt2"] = self.period[i]["T2aHE"] - self.period[i]["T2bHE"]

    @property
    def period(self) -> Dict[pd.core.frame.DataFrame]:
        return self._period

    @property
    def period_keys(self) -> List[str, float, int]:
        return self._period.keys()

    @period.setter
    def period(self, val: Dict[pd.core.frame.DataFrame]) -> None:
        self._period = val
        self._period_keys = val.keys()

    def _nan_handler(
        self, 
        period: Dict[pd.core.frame.DataFrame]
    ) -> Dict[pd.core.frame.DataFrame]:
        '''
        #* dictionary grouped by time period may
        #* include many rows with NaN values due to
        #* grouping by small periods like minutes (secs and etc.) 
        #* Parameters
        #* ----------
        #* period: Dict[pd.core.frame.DataFrame]
        #* Returns
        #* ----------
        #* filtered dict - NaN cols/rows filled 
        #* by zeros for each df
        '''
        
        for i in period.keys():
            period[i] = period[i].fillna(0.0) 

        return period

    def selection_validator(func: Callable):
        '''
        #* Decorator to check if columns
        #* to select are among df columns
        #* to make sure wrapper does everething
        #* <columns> argument always must be at first place
        #* Parameters
        #* ----------
        #* func: Callable
        #*  function that was called
        #* Returns
        #* ----------
        #* wrapper
        '''
        def wrapper(self, *args, **kwargs):
            '''
            #* check if columns are in df
            #* if passes calls inital func
            #* Parameters
            #* ----------
            #* args/kwargs are arguments passed
            #* to initially called func
            #* Raises
            #* ----------
            #* KeyError
            #*  if columns are not among df columns
            #* Returns
            #* ----------
            #* None
            '''

            try:
                #* getting first element in agrs
                #* if another string is passed
                #* KeyError will trigger after checking the columns names
                cols = args[0]
            except IndexError:
                cols = kwargs.get("column") if not kwargs.get("columns")\
                    else kwargs.get("columns")
                print(cols)
                if cols is None:
                    raise KeyError(
                        "columns argument not given either as args or kwargs"
                    )

            if not isinstance(cols, list):
                cols = [cols]
            
            cols = set(cols)

            for i, v in self.period_items:
                if not cols.issubset(set(v.columns)):
                    raise KeyError(
                        f"some of a given columns are not in dataframe for {i}"
                    )
            
            return func(self, *args, **kwargs)
        return wrapper
        
    def sma_smoothing(
        self,
        num_points: int
    ) -> Dict[str, pd.core.dataframe.DataFrame]:
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
        #* copy of df
        '''
        st: int = 0
        fn:int = num_points
        for i in self.period_keys:
            rows, cols = len(self.period[i]), len(self.period[i].columns)
            arr = np.array([])

            for n in range(len(self.period[i])):
                #* select rows according to number of points to get sma and getting mean in each column
                arr = np.append(arr, self.period[i].iloc[st + n : fn + n].mean(axis=0))
            
            #* reshape to 2D array
            arr = arr.reshape(rows, cols) #[]
            #* repcale previos values by creating new df
            self.period[i] = pd.DataFrame(
                data=arr,
                index=self.period[i].index,
                columns=self.period[i].columns)   
            #* cut new df by num_points
            self.period[i] = self.period[i].iloc[:-num_points] 
            # print(arr, arr.shape, self.period[i].shape)
        return self.period

    @selection_validator
    def conditional_rows_drop(
        self,
        columns: list,
        operator: str,
        value: int | float,
        # fillna: float | int | str = 0.0, #todo add
        # period = None

    ) -> Dict[str, pd.core.framew.DataFrame]:
        '''
        #* the method drops rows where condition is true
        #* for all columns provided to cols variable
        #* Parameters
        #* ----------
        #* cols: str | list
        #*  columns to apply conditions on
        #* Returns
        #* ----------
        #* None
        '''
        #todo any thoughts on how to handle it?
        # if period is not None:

        
        for i in self.period_keys:
            to_drop = np.array([])
            self.period[i].loc[:, columns] = self.period[i].loc[:, columns].fillna(0.0)
            to_filter = self.period[i].loc[:, columns]

            rows = (i for i in to_filter.index)
            for k in rows:
                row = to_filter.loc[k, :]
                method = self._get_binary_operator(row, operator)
                res = method(row, value).values
                if not False in res:
                    to_drop = np.append(to_drop, k)
                    # print(res, "row to drop", k)

            self.period[i] = self.period[i].drop(index=to_drop)
        return self.period

    @selection_validator
    def pumps_mapping(
        self,
        columns: List[str] = ["p21", "p22", "p23", "p24"],
        drop_pumps: bool = False
    ) -> None:
        '''
        #* To maintain reactor under normal condition
        #* there are no need to operate all pumps at
        #* the same time
        #* However, all pumps influence differently on
        #* water flow intensity in a circuit
        #* So this method aim is to make new feature
        #* shows what are pumps that are currently under operation 
        #* Parameters
        #* ----------
        #* pumps_columns: List[str]
        #*  array of columns to select from df
        #* drop pumps: bool = True
        #*  boolean to drop pumps_column from df
        #* Returns
        #* ----------
        #* modified dict
        '''
        #todo rewrite method to check rows sequentially
        #todo it may helps to catch a moment of pumps replacing with each other

        opmv = self.OPMV_MAP.get("Ipumps")
        

        for i in self.period_keys:

            pumps_names = np.array([])
            # self.period[i]["pumps2"] = np.zeros_like(len(self.period[i]))
            
            #* select columns
            cols = self.period[i].loc[:, columns]
            
            for j in range(len(cols)):
                #* puo stands for PumpsUnderOpereration
                puo_ind = cols.iloc[j][cols.iloc[j] > opmv].index
                
                if len(puo_ind) == 0:
                    puo = np.nan
                else:    
                    puo = "".join(
                        [self.PUMPS_MAP.get(k) for k in puo_ind]
                    )

                pumps_names = np.append(pumps_names, puo)
            
            #* add and populate new feature
            self.period[i]["pumps2"] = pd.Series(pumps_names).values
        
        if drop_pumps:
            self.period[i] = self.period[i].drop(columns, axis=1)
            
        return self.period

    #! create z-score test
    @selection_validator
    def filter_by_deviation(
        self, 
        column: str | float | int, 
        byvalue: float | int | None = None,
        bysigma: int | None = None,  #* 2 sigma == 2 std.dev -> capture 95% of data
    ) -> None:
        '''
        #* 
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
        if byvalue and bysigma:
            raise ValueError(
                f"Both byvalue: {byvalue} and bysigma: {bysigma} are given. Choose one"
            )

        for i in self.period_keys:
            
            no_zeros = self.period[i].loc[:, column][self.period[i].loc[:, column] > 0]
            no_zeros_mean = no_zeros.mean()

            if byvalue:
                lower, upper = no_zeros_mean - byvalue, no_zeros_mean + byvalue
                
            
            elif bysigma:
                std = np.std(no_zeros)
                sigma = bysigma*std
                lower, upper = no_zeros_mean - sigma, no_zeros_mean + sigma

            # print(lower, no_zeros_mean, upper)
            self.period[i] = self.period[i][
                    (self.period[i][column] < upper)
                    & (self.period[i][column] > lower)
                ]
            
        return self.period


    def filter_by_zscore(
        self,
        column: str | float | int,
        pvalue: float = 0.95
    ):
        '''
        #* Approach to detect and remove anomalies from dataset
        #* assuming dataset alike normal distribution
        #* zscores uses to standartize data by mean and stdev as follows:
        #* Z = (x - mean) / stdev
        #* the Z results can be interpret as:
        #* how far values lies beyond the mean
        #* to classify Z to normal / anomaly the are confidence intervals
        #* Example: pvalue = 0.95 -> critical value is +-1.96, so
        #* if Z-value beyond +-1.96 it's an anomaly
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
        

        

        lower, upper = self.CRITICAL_VALUES.get(pvalue)
        opmv = self.OPMV_MAP.get(column)

        for i in self.period_keys:
            
            #* OnPower Minimum Value
            self.period[i] = self.period[i][self.period[i].loc[:, column] > opmv] 
            standartized = pd.Series(
                stats.zscore(self.period[i][column])
            )
            

            normal_index = standartized[
                (standartized < upper)
                & (standartized > lower)
            ].index

            #* drops rows where the value beyond the critical value 
            self.period[i] = self.period[i].loc[normal_index, :]
            # self.period[i].index = 
        return self.period

    def to_dataframe(self):
        df = pd.concat(self.period.values())
        df.index.name = "Timestamp"
        return df


class FeatureSelection:
    def __init__(self):
        return

class FeatureEngineering(DataPreprocess):
    # OPERATORS_MAP = {

    # }

    def __init__(
        self,
        df: pd.core.frame.DataFrame,
        default_features: bool = True
    ):
        super().__init__()
        self._df = df.fillna(0.0)

        if default_features:
            self._make_default_features()

    @property
    def df(self) -> pd.core.frame.DataFrame:
        return self._df

    @df.setter
    def df(
        self, 
        val: pd.core.frame.DataFrame
    ) -> None:
        self._df = val

    def _make_default_features(self) -> None:
        
        self.df["dt_circuits_coef"] = self.df["T1bHE"]/self.df["T2bHE"]
        self.df["dt_circuits_coef_delta"] = np.absolute(self.df["T1bHE"] - self.df["T2bHE"])

    # def _gt(self, series, val, i, e):
    #     return np.where(
    #         series > val,
    #         i, #* if condition return true 
    #         e  #* if condition return false 
    #     )


    def selection_validator(func: Callable):
        '''
        #* Decorator to check if columns
        #* to select are among df columns
        #* to make sure wrapper does everething
        #* <columns> argument always must be at first place
        #* Parameters
        #* ----------
        #* func: Callable
        #*  function that was called
        #* Returns
        #* ----------
        #* wrapper
        '''
        def wrapper(self, *args, **kwargs):
            '''
            #* check if columns are in df
            #* if passes calls inital func
            #* Parameters
            #* ----------
            #* args/kwargs are arguments passed
            #* to initially called func
            #* Raises
            #* ----------
            #* KeyError
            #*  if columns are not among df columns
            #* Returns
            #* ----------
            #* function result
            '''
            try:
                #* getting first element in agrs
                #* if another string is passed
                #* KeyError will trigger after checking the columns names
                cols = args[0]
            except IndexError:
                cols = kwargs.get("column") if not kwargs.get("columns")\
                    else kwargs.get("columns")
                print(cols)
                if cols is None:
                    raise KeyError(
                        "columns argument not given either as args or kwargs"
                    )

            if not isinstance(cols, list):
                cols = [cols]
            
            cols = set(cols)

            if not cols.issubset(set(self.df.columns)):
                raise KeyError(
                    f"some of a given columns are not in dataframe"
                )
        
            return func(self, *args, **kwargs)
        return wrapper

    def columns_categorizing(
        self,
        columns: list,
        value: float | int,
        i: str | float | int,
        e: str | float | int |  None = None, #! when none results are wrong
        operator: str = "gt",
    ):
        '''
        #* Categorizing 1/0 aka if/else by
        #* conditional filtering
        #* Parameters
        #* ----------
        #* columns: list
        #*  columns to apply binary conditions on
        #* value: float | int
        #*  value uses in condition
        #* i: str | float | int
        #*  replaces row value of column 
        #*  if condition is True
        #* e: str | float | int
        #*  replaces row value of column 
        #*  if condition is False
        #* operator: str
        #*  binary operator to apply as a condition on data
        #* Returns
        #* ----------
        #* copy of df
        '''
    
        method = self._get_binary_operator(self.df.loc[:, columns[0]], operator)

        for c in columns:
            
            if e is None:
                # e = self.df[c]  #* inital value of column
                self.df[c] = np.where(
                    method(self.df[c], value),
                    i,
                    self.df[c]
                )
            else:
                self.df[c] = np.where(
                        method(self.df[c], value),
                        i,
                        e
                    )

        return self.df
    
    @selection_validator
    def filter_by_deviation(
        self, 
        column: str | float | int, 
        value: float | int = 0.1,
        include_zeros: bool = False
    ) -> None:
        
        
        # print(i)
        #* include zeros in column mean
        if include_zeros:
            col_mean = self.df.loc[:, column].mean()
        else:
            col_mean = self.df.loc[:, column][self.df.loc[:, column] > 0].mean()

        self.df = self.df[np.absolute(1 - self.df.loc[:, column]/col_mean) <= value]
            
        return self.df

    #todo add feature to filter multiple columns
    @selection_validator
    def filter_by_zscore(
        self,
        column: str | float | int,
        pvalue: float = 0.95,
        nan_policy="omit"
    ):
        '''
        #* Approach to detect and remove anomalies from dataset
        #* assuming dataset alike normal distribution
        #* zscores uses to standartize data by mean and stdev as follows:
        #* Z = (x - mean) / stdev
        #* the Z results can be interpret as:
        #* how far values lies beyond the mean
        #* to classify Z to normal / anomaly the are confidence intervals
        #* Example: pvalue = 0.95 -> critical value is +-1.96, so
        #* if Z-value beyond +-1.96 it's an anomaly
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

        lower, upper = self.CRITICAL_VALUES.get(pvalue)
        
        standartized = pd.Series(
            #todo sensetive to nan -> add decorator to inform
            stats.zscore(self.df[column], nan_policy=nan_policy)
        )
        

        normal_index = standartized[
            (standartized < upper)
            & (standartized > lower)
        ].index

        #* drops rows where the value beyond the critical value 
        self.df = self.df.loc[normal_index, :]
            # self.period[i].index = 
        return self.df

    @selection_validator
    def conditional_rows_drop(
        self,
        columns: list,
        operator: str,
        value: int | float,
        # fillna: float | int | str = 0.0, #todo add
        # period = None

    ) -> pd.core.framew.DataFrame:
        '''
        #* the method drops rows where condition is true
        #* for all columns provided to cols variable
        #* Parameters
        #* ----------
        #* cols: str | list
        #*  columns to apply conditions on
        #* Returns
        #* ----------
        #* None
        '''
        
        to_drop = np.array([])
        to_filter = self.df.loc[:, columns]
        rows = (i for i in to_filter.index)

        for k in rows:
            row = to_filter.loc[k, :]
            method = self._get_binary_operator(row, operator)
            res = method(row, value).values
            if not False in res:
                to_drop = np.append(to_drop, k)
                # print(res, "row to drop", k)
        self.df = self.df.drop(index=to_drop)

        return self.df


    @selection_validator
    def columns_averaging(
        self,
        columns: List[str],
        omitbelow: bool | float | int = 100,
        feature_name: str = "new"
    ) -> pd.core.dataframe.DataFrame:
        '''
        #* creates new feature by avereging
        #* given columns for each row. If value in a column
        #* lower than <omitbelow> value
        #* removes column from averaging
        #* Parameters
        #* ----------
        #* columns: list
        #* columns of df to be averaged
        #* Returns
        #* ----------
        #* dataframe
        '''
        aim = self.df.loc[:, columns]
        gen = (i for i in range(len(aim)))
        store = np.array([])
        for i in gen:
            store = np.append(store, aim.iloc[i][aim.iloc[i] > omitbelow].mean())
        
        self.df[feature_name] = pd.Series(store).values
        return self.df

    def make_time_onpower_feature(
        self,
        start: float | int = 0,
        byindex: bool = True,
        bycolumn: bool = False,
        column_name: str | None = "Timestamp",
        time_periods: List[Tuple[str, str]] | None = None,
        unit: str = "hour",  #* under dev
        feature_name: str = "new"

    ):
        '''
        #* Under the hood of this method
        #* lies the idea to create new feature
        #* that indicates the duration (time) onpower
        #* basicly feature shows for how long time smth
        #* is under operation. Method computates
        #* the time from TimeStamp column/ index in
        #* different units (hour by default)
        #* Parameters
        #* ----------
        #* start: float | int
        #*  the start value to increment to
        #* byindex: bool
        #*  indicates that TimeStamp is index
        #* bycolumm: bool
        #*  indicates that TшmeStamp is column
        #* column_name: str
        #*  column name of Timestamp
        #* time_period: List[Tuple[str, str]]
        #*  uses to restrict Timestamp boundaries
        #*  and computate time onpower only in a given
        #*  periods
        #* unit: str
        #*  output unit of camputated time onpower
        #* Raises
        #* ----------
        #* ValueError
        #*  if some method arguments were given
        #*  in a wrong way
        #* Returns
        #* ----------
        #*
        '''
        # self.df = self.df.loc["2022-10-17":"2022-11-18", :]

        if byindex and bycolumn:
            raise ValueError(
                f"byindex and bycolumn both set to True"
            )

        if bycolumn and not column_name:
            raise ValueError(
                f"column_name is None"
            )

        tot_time = 0.0
        length = len(self.df)

        if time_periods:
            #! needs only to get the correct timedelta value -> hours
            #! but computated hours will be devided by a total length of df

            #todo all data before and after first and last dates drops
            
            self.df[feature_name] = np.zeros(length)

            # self.df[feature_name] = 
            for i in time_periods:
                st, fn = i  #* unpacking of dates tuple
                period = self.df.loc[st: fn, feature_name]

                length = len(period)
                
                tot_time_diff = pd.to_datetime(fn) - pd.to_datetime(st)
                tot_time = self.get_hours(tot_time_diff)

                feature = np.arange(
                    start + tot_time/length,
                    start + tot_time + tot_time/length,
                    tot_time/length
                )
                print(feature, len(feature), len(self.df.loc[st: fn, feature_name]))
                if len(pd.Series(feature).values) > len(self.df.loc[st: fn, feature_name]):
                    self.df.loc[st: fn, feature_name] = pd.Series(feature).values[: len(self.df.loc[st: fn, feature_name])]
                    return self.df
                
                self.df.loc[st: fn, feature_name] = pd.Series(feature).values

        
        else:

            tot_time_diff = pd.to_datetime(self.df.index[-1]) - pd.to_datetime(self.df.index[0])
            tot_time = self.get_hours(tot_time_diff)

            feature = np.arange(
                start + tot_time/length,
                start + tot_time + tot_time/length,
                tot_time/length
            )

            #* create a new feature and add it to df
            self.df[feature_name] = pd.Series(feature).values

        # print(tot_time, feature, feature.shape, length)

        return self.df

    #todo make default_fe
    def make_dt2(self):
        # todo decorator - validator requires
        '''
        #* This is built-in method to get
        #* temperature deffirence on 2nd circiuit
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
        
        return self.df["T2aHE"] - self.df["T2bHE"]

   
    #todo make default_fe
    def make_dt1(self):
        # todo decorator validator requires
        '''
        #* This is built-in method to get
        #* temperature deffirence on 2nd circiuit
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
        
        return self.df["T1bHE"] - self.df["T1aHE"]

    def pumps_normalizer(
        self,
        column: str
    ):
        '''
        #* Normalize <column_name> column
        #* by a coefficient depending on
        #* current pumps under operation
        #* Parameters
        #* ----------
        #* column_name: str
        #*  column of df to apply normalization on
        #* Returns
        #* ----------
        #* normalized copy of column
        '''
        #todo make pumps2 default name and changeble


        self.df.loc[:, column] = np.where(
            self.df.loc[:, "pumps2"] == "124",
            self.df.loc[:, column]*1.02,
            np.where(
                self.df.loc[:, "pumps2"] == "123",
                self.df.loc[:, column]*1.03,
                self.df.loc[:, column]
            )
            
        )
        return self.df

    def make_QbyIP(self):
        self.df["QbyIP"] = self.df["Q2"]/(self.df["P2"]*self.df["I2mean"])
       
        return self.df

    def make_QbyI(self):
        self.df["QbyIP"] = self.df["Q2"]/(self.df["P2"]*self.df["I2mean"])
        return self.df

    def make_dts_on_HEs(
        self, 
        inplace=True, 
        norm_by_delta: bool | None = True,
        norm_by_rel: bool | None = None
    ) -> pd.core.dataframe.DataFrame:
        '''
        #* computation of dt on each HE
        #* if inplace set to True replace
        #* existing values by dt
        #* if inplace set to False creates new features
        #* Parameters
        #* ----------
        #* inplace: bool
        #*  var to make or not inplace replacement
        #* Returns
        #* ----------
        #* df
        '''

        #! add exception on norm_by_delta and norm_by_rel uses

        #* default names of columns
        he_temps = ["T2aHE1", "T2aHE2", "T2aHE3", "T2aHE4", "T2aHE5"]

        dts = self.df.loc[
                :, 
                he_temps
            ].apply(lambda x: x - self.df["T2bHE"])

        if not inplace:
            #* new features 
            he_temps = np.array(list(map(lambda x: f"d{x}", he_temps)))

        if norm_by_rel:
            dts = dts.apply(lambda x: x / self.df["dt_circuits_coef"])

        if norm_by_delta:
            dts = dts.apply(lambda x: x / self.df["dt_circuits_coef_delta"])

        self.df.loc[:, he_temps] = dts
        
        return self.df

    def make_heat_dissipation(
        self,
        feature_name: str ="Ndis",
        dt_norm: bool = True
    ) -> pd.core.dataframe.DataFrame:
        '''
        #* It's like heat_dissipation computation
        #* but the aim is to show relation between
        #* Q - wfr and dt
        #* However, in practice dt between hot water of 1st circuit
        #* and cold water of 2nd circuit must be taken into account
        #* because as higher difference between this two as higher dt on HE
        #* this is effect of temparature gradient between two circuits
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
        self.df[feature_name] = self.df["QbyIP"]*self.df["dt2"]/self.df["dt_circuits_coef"]

        if not dt_norm:
            self.df[feature_name] = self.df["QbyIP"]*self.df["dt2"]


        return self.df

    #todo make it for inside uses only
    def get_hours(self, timedelta):
        days = timedelta.days
        seconds = timedelta.seconds
        return days*24 + seconds/3600

