from __future__ import annotations
from typing import Dict, List, Union, Callable, TypeVar, Generic, Tuple
from sklearn.model_selection import cross_validate
import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import HistGradientBoostingRegressor,\
    GradientBoostingRegressor, AdaBoostRegressor, RandomForestRegressor

__all__ = ['SingleStepOutput', 'MultiStepOutput', 'Model']

T = TypeVar('T')
CV = Tuple[
    str,  #* name of resampler
    Union[int, float],  #* configuration
    bool  #* returns indices if True, splitted data if it's False
]


class Model(Generic[T]):
    def __init__(
        self,
        X: Generic[T],
        y: Generic[T]
    ) -> None:
        self.X = X
        self.y = y

    def model_comparison(self, model_list: List[str],
            models: List[object], cv: object) -> None:
        for i in range(len(models)):
            print(f"Evaluating of {model_list[i]}")
            self.cross_validation(models[i], cv)

    def cross_validation(
            self,
            model: object,
            cv: object,
            X: Union[np.ndarray, pd.core.frame.DataFrame] = None,
            y: Union[np.ndarray, pd.core.frame.DataFrame] = None
            ) -> None:

        if X is None or y is None:
            X = self.X
            y = self.y

        cv_results = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=["neg_mean_absolute_error", "neg_root_mean_squared_error"]
        )
        mae = -cv_results["test_neg_mean_absolute_error"]
        rmse = -cv_results["test_neg_root_mean_squared_error"]
        print(
            # f"Model: {model.__name__}\n"
            f"Mean Absolute Error:     {mae.mean():.3f} +/- {mae.std():.3f}\n"
            f"Root Mean Squared Error: {rmse.mean():.3f} +/- {rmse.std():.3f}\n"
        )

    
class TimeSeriesForecast(Model, Generic[T]):
    RESAMPLERS = {
        "tss": "time_series_split",
        "TimeSeriesSplit": "time_series_split",
        "default": "default_split"
    }

    MODELS = {
        "HGBR": {
            "name": "HistGradientBoostingRegressor",
            "grid": {
                "n_estimators": [10, 50, 100, 500],
                "learning_rate": [0.0001, 0.001, 0.01, 0.1, 1.0],
                "subsample": [0.5, 0.7, 1.0],
                "max_depth": [3, 7, 9]
            }
        },
        "GBR": {
            "name": "GradientBoostingRegressor",
            "grid": {
                "n_estimators": [10, 50, 100, 500],
                "learning_rate": [0.0001, 0.001, 0.01, 0.1, 1.0],
                "subsample": [0.5, 0.7, 1.0],
                "max_depth": [3, 7, 9]
            }
        },
        "RFR": {
            "name": "RandomForestRegressor",
            "grid": {
                "n_estimators": [10, 50, 100, 500],
                "max_depth": [3, 7, 9]
            }
        }
    }

    #* the parent class consists of
    #* main pattern to handle TimeSeries dataset
    def __init__(
        self,
        X: Generic[T],
        y: Generic[T],
        #* TSsplit of default 80/20 split
        #* (name of resampler, its configuration) = (tss, n_splits) 
        cv_config: CV = ("tss", 5, True)
    ) -> None:
        super().__init__(X, y)
        
        self.cv = self._resample_data(cv_config)
        # self.train, self.test

    def _get_resample_method(
        self, 
        resampler_name: str
    ) -> Callable:
        method = self.RESAMPLERS.get(resampler_name)
        if method is None:
            raise KeyError("There are no given resampler method")
        return getattr(self, method)  #* returns callable 

    def _resample_data(self, cv_config: tuple):
        method, config, indices = cv_config
        #* now it's callable
        method = self._get_resample_method(method)
        res = method(config, indices)
        return res
     
    def time_series_split(
        self, 
        n_splits:int,
        indices: bool = True
    ) -> List[Generic[T]]:
        #* 2d array of indices
        if not indices:
            #* do smth
            return
        
        tss = list(
            TimeSeriesSplit(n_splits=n_splits).split(self.X, self.y)
        )

        return tss

    def default_split(
        self, 
        train_size: float = 0.8,
        indices: bool = True
    ) -> List[Generic[T]]:
        '''
        #* Makes default split (like 80/20)
        #* Parameters
        #* ----------
        #* train_size: float
        #*  size of train dataset
        #* Returns
        #* ----------
        #* 2d array consists of training 
        #* and test indices
        '''
        test_size = 1 - train_size
        train_length = int(len(self.X)*train_size)
        # test_length = int(len(self.X)*test_size)
        length = len(self.X)
        # train_X = self.X[:train_length]
        train_X = self.X[ :train_length]
        train_y = self.y[ :train_length]

        test_X = self.X[train_length: ]
        test_y = self.y[train_length: ]
        
        if indices:
            train = self._make_indices(0, train_length)
            test = self._make_indices(train_length, length)
            
            return [train, test]

        return [
            (train_X, train_y),
            (test_X, test_y)
        ]
    
    def _make_indices(
        self,
        st: int,
        fn: int
    ):
        return np.arange(st, fn, 1)
    
    def make_model(
        self,
        model: str | None = None
    ):
        return
    
    def _get_model(self, name):

        return

    def HistGradientBoostingRegressor(
        self,
        **params: dict
    ):
        return 

class SingleStepOutput(TimeSeriesForecast):
    def __init__(
        self,   
        X: Generic[T],
        y: Generic[T],
        cv_config: CV = ("tss", 5, True)
    ) -> None:
        super().__init__(X=X, y=y, cv_config=cv_config)
        # self.X = self.cv
    
    def forecast(self):
        
        return

class MultiStepOutput(TimeSeriesForecast):
    def __init__(self):
        return

    def recursive_forecast(self):
        return
