from __future__ import annotations
from calendar import leapdays
import re
from xml.etree.ElementInclude import include
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Tuple, Union, Set
from collections import defaultdict


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
    
    def __init__(self):
        return

    @abstractmethod
    def conditional_rows_drop(self):
        raise NotImplementedError

    @abstractmethod
    def filter_by_deviation(self):
        raise NotImplementedError

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
        period: Dict[pd.core.frame.DataFrame]
    ) -> None:

        super().__init__()
        self._period = self._nan_handler(period)
        self.period_items = (i for i in self.period.items())
        self._period_keys = self.period.keys()

        
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
        

    @selection_validator
    def conditional_rows_drop(
        self,
        columns: list,
        condition: str,
        value: int | float,
        # fillna: float | int | str = 0.0, #todo add
        # period = None

    ) -> Dict[pd.core.framew.DataFrame]:
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
                method = getattr(type(row), condition)
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
        
        for i in self.period_keys:

            cols = self.period[i].loc[:, columns]
            #* getting mean of each column
            cols_mean = cols.fillna(0.0).mean(axis=0)

            #* PumpsUnderOpereration
            puo_ind = cols_mean[cols_mean > 80].index

            puo = "".join(
                [self.PUMPS_MAP.get(i) for i in puo_ind]
            )

            #* modify df
            self.period[i]["pumps2"] = pd.Series(
                np.full(
                    (len(cols), ),
                    puo
                )
            ).values

            if drop_pumps:
                self.period[i] = self.period[i].drop(columns, axis=1)
            
        return self.period

    @selection_validator
    def filter_by_deviation(
        self, 
        column: str | float | int, 
        value: float | int = 0.1,
        include_zeros: bool = False
    ) -> None:
        
        for i in self.period_keys:
            # print(i)
            #* include zeros in column mean
            if include_zeros:
                col_mean = self.period[i].loc[:, column].mean()
            else:
                col_mean = self.period[i].loc[:, column][self.period[i].loc[:, column] > 0].mean()

            self.period[i] = self.period[i][np.absolute(1 - self.period[i][column]/col_mean) <= value]
            
        return self.period

    def to_dataframe(self):
        return pd.concat(self.period.values())


class FeatureSelection:
    def __init__(self):
        return

class FeatureEngineering(DataPreprocess):

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
        self.df["dt1"] = self.df["T1bHE"] - self.df["T1aHE"]
        self.df["dt2"] = self.df["T2aHE"] - self.df["T2bHE"]
        self.df["dt_circuits_coef"] = self.df["T1bHE"]/self.df["T2bHE"]

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

    @selection_validator
    def conditional_rows_drop(
        self,
        columns: list,
        condition: str,
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
            method = getattr(type(row), condition)
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
                
                tot_time_diff = pd.to_datetime(period.index[-1]) - pd.to_datetime(period.index[0])
                tot_time = self.get_hours(tot_time_diff)

                feature = np.arange(
                    start + tot_time/length,
                    start + tot_time + tot_time/length,
                    tot_time/length
                )

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
        dt_norm: bool=True
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

        #* default names of columns
        he_temps = ["T2aHE1", "T2aHE2", "T2aHE3", "T2aHE4", "T2aHE5"]

        dts = self.df.loc[
                :, 
                he_temps
            ].apply(lambda x: x - self.df["T2bHE"])

        if not inplace:
            #* new features 
            he_temps = np.array(list(map(lambda x: f"d{x}", he_temps)))

        if dt_norm:
            dts = dts.apply(lambda x: x / self.df["dt_circuits_coef"])

        self.df.loc[:, he_temps] = dts
        
        return self.df

    def make_heat_dissipation(
        self, 
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
        self.df["Ndis"] = self.df["QbyIP"]*self.df["dt2"]/self.df["dt_circuits_coef"]

        if not dt_norm:
            self.df["Ndis"] = self.df["QbyIP"]*self.df["dt2"]


        return self.df

    #todo make it for inside uses only
    def get_hours(self, timedelta):
        days = timedelta.days
        seconds = timedelta.seconds
        return days*24 + seconds/3600

