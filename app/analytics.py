from abc import ABC, abstractmethod
from gettext import find
from typing import List, Tuple, Union
import numpy as np
import pandas as pd
from sklearn import linear_model
from scipy.signal import find_peaks, peak_widths

__all__ = ["WaterTemperatures", "WaterFlowRates", "DissipatedHeat", "PumpsCurrents", "FansCurrents"]

class Analytics(ABC):
    def __init__(self, mean_deviation_lim: Union[float, int, None] = None) -> None:
        self._mean_deviation_lim: Union[float, int, None] = mean_deviation_lim

    @property
    def mean_deviation_lim(self) -> Union[float, int, None]:
        return self._mean_deviation_lim

    @mean_deviation_lim.setter
    def mean_deviation_lim(self, value: Union[float, int]) -> None:
        self._mean_deviation_lim = value
        return None

    def lin_regression_fit(self, X: np.ndarray, y: np.ndarray) -> linear_model.LinearRegression:
        """ fit given data to linear regression model
            returns trained model"""
        lr: linear_model.LinearRegression = linear_model.LinearRegression() 
        return lr.fit(X, y)

    @abstractmethod
    def df_md_filter(self, df: pd.core.frame.DataFrame, col_name: Union[int, str]):
        """ 
        *filter df by mean deviation limit
        *col_name denotes the column to filter whole df
        *returns 2 dfs: ones that are True and False
        """

        if self.mean_deviation_lim is None:
            raise ValueError("mean deviation limit is not given")
        pass


class WaterTemperatures(Analytics):
    def __init__(self, mean_deviation_lim: Union[float, int, None] = None) -> None:
        super().__init__(mean_deviation_lim)

    def df_md_filter(self, df: pd.core.frame.DataFrame, 
            col_name: Union[int, str]) -> Tuple[pd.core.frame.DataFrame, pd.core.frame.DataFrame]:
        
        col_mean = df[col_name].mean()
        df_true = df[np.absolute(df[col_name] - col_mean) < self.mean_deviation_lim]
        df_false = df[np.absolute(df[col_name] - col_mean) > self.mean_deviation_lim]
        return df_true, df_false

class WaterFlowRates(Analytics):
    def __init__(self, mean_deviation_lim: Union[float, int, None] = None) -> None:
        super().__init__(mean_deviation_lim)

    def df_md_filter(self, df: pd.core.frame.DataFrame, 
            col_name: Union[int, str]) -> Tuple[pd.core.frame.DataFrame, pd.core.frame.DataFrame]:
    
        col_mean = df[col_name].mean()
        df_true = df[np.absolute(1 - (df[col_name]/col_mean)) < self.mean_deviation_lim]
        df_false = df[np.absolute(1 - (df[col_name]/col_mean)) > self.mean_deviation_lim]
        return df_true, df_false

class DissipatedHeat(Analytics):
    def __init__(self, mean_deviation_lim: Union[float, int, None] = None) -> None:
        super().__init__(mean_deviation_lim)

    #* filter by taking portion of col_mean 
    def df_md_filter(self, df: pd.core.frame.DataFrame, 
            col_name: Union[int, str]) -> Tuple[pd.core.frame.DataFrame, pd.core.frame.DataFrame]:
    
        col_mean = df[col_name].mean()
        df_true = df[np.absolute(1 - (df[col_name]/col_mean)) < self.mean_deviation_lim]
        df_false = df[np.absolute(1 - (df[col_name]/col_mean)) > self.mean_deviation_lim]
        return df_true, df_false

class PumpsCurrents(Analytics):
    def __init__(self, mean_deviation_lim: Union[float, int, None] = None) -> None:
        super().__init__(mean_deviation_lim)

    def df_md_filter(self, df: pd.core.frame.DataFrame, 
            col_name: Union[int, str]) -> Tuple[pd.core.frame.DataFrame, pd.core.frame.DataFrame]:
    
        col_mean = df[col_name].mean()
        df_true = df[np.absolute(1 - (df[col_name]/col_mean)) < self.mean_deviation_lim]
        df_false = df[np.absolute(1 - (df[col_name]/col_mean)) > self.mean_deviation_lim]
        return df_true, df_false


class FansCurrents(Analytics):
    def __init__(self, mean_deviation_lim: Union[float, int, None] = None) -> None:
        super().__init__(mean_deviation_lim)

    @staticmethod
    def detect_nonstationarity(y: pd.core.frame.Series,
             operation_threshold: Union[float, int]):
        """
        * applicable only for data that considers as binary (0 - off, 1 -on)
        """
        return