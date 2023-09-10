from __future__ import annotations
from posixpath import split
from typing import Dict, List, Union, Callable, TypeVar, Generic, Tuple
from sklearn.model_selection import cross_validate
import numpy as np
import pandas as pd
import sklearn

from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    GradientBoostingRegressor, 
    AdaBoostRegressor, 
    RandomForestRegressor
)

from sklearn.compose import ColumnTransformer


__all__ = ['SingleStepOutput', 'MultiStepOutput', 'Model', "Preprocessing"]

T = TypeVar('T')
CV = Tuple[
    str,  #* name of resampler
    Union[int, float],  #* configuration
    bool  #* returns indices if True, splitted data if it's False
]


class Preprocessing(Generic[T]):
    '''
    #* This class serves to
    #* make preprocessing of dataset
    #* before apply machine learning model
    #* on it
    #* Attributes
    #* ----------
    #*
    #* Methods
    #* ----------
    #*
    '''
    def __init__(
        self,
        X: Generic[T] #* pandas df
    ):
        self.X = X
        self.deep_X = X.copy()
        self.dump_cols = X.columns
        self.preserve_X = None

    def add_shifts(
        self,
        columns: list,
        lags: int,
        reset_index: bool = False
    ):
        self.preserve_X = self.X.copy() #* stores real values
        for i in columns:
            for j in range(1, lags+1):
                self.X.loc[:, f"{i}_lag_{j}"] = self.X.loc[:, i].shift(j)

        self.X = self.X.dropna(axis=0)
        
        if reset_index:
            
            self.preserve_X = self.preserve_X.loc[self.X.index, :] #* getting data by index of modified X
            self.X = self.X.reset_index().drop("index", axis=1) #* reindexing
            self.preserve_X = self.preserve_X.reset_index().drop("index", axis=1) #* reindexing
            
        return self.X
    
    def column_transformer(
        self,
        output: str | None = "pandas" ,
        **kwargs  #* ColumnTransformer parameters
    ) -> None:
        ct = ColumnTransformer(
         **kwargs   
        )

        if output:
            ct = ct.set_output(transform="pandas")

        ct.fit(self.X)
        self.X = ct.transform(self.X) #* rewrites X
        return self.X

    


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
        self.train_X, self.train_y  = None, None
        self.test_X, self.test_y  = None, None
        self.cv = self._resample_data(cv_config)
        self.model = None
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
        indices: bool = True,

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
        self.train_X = self.X[ :train_length]
        self.train_y = self.y[ :train_length]

        self.test_X = self.X[train_length: ]
        self.test_y = self.y[train_length: ]
        
        if indices:
            train = self._make_indices(0, train_length)
            test = self._make_indices(train_length, length)
            
            return [
                (train, test)
            ]

        return [
            (self.train_X, self.train_y),
            (self.test_X, self.test_y)
        ]
    
    def _make_indices(
        self,
        st: int,
        fn: int
    ):
        return np.arange(st, fn, 1)
    
    def _model_fit(
        self,
        model_name: str,
    ) -> None:
        model_str = self.MODELS.get(model_name).get("name")
        print(model_str)
        model = getattr(sklearn.ensemble, model_str)

        if model is not None:
            #* fitting
            self.model = model().fit(
                self.train_X,
                self.train_y
            )
            return None
            
        raise KeyError(f"Given model name {model_name} does not exist")

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
    
    def default_split(
        self, 
        train_size: float = 0.8, 
        indices: bool = True
    ) -> List[Generic[T]]:
        #* modifing splits to sso
        splits = super().default_split(train_size, indices)
        
        if len(splits) == 1:
            train, test = splits[0]
            test = np.array([test[0]])
            
            self.train_X, self.train_y = self.X[:len(train)], self.y[:len(train)]
            self.test_X, self.test_y = self.X[len(train): len(train) + len(test)], self.y[len(train): len(train) + len(test)]

            return [
                (train, test)
            ]
            
        train_X, train_y = splits[0]
        test_X, test_y = splits[1]

        test_X, test_y = [test_X[:1]], [test_y[:1]]
        
        self.train_X, self.train_y = train_X, train_y
        self.test_X, self.test_y = test_X, test_y

        return [
            (train_X, train_y),
            (test_X, test_y)
        ]


    def forecast(
        self,
        model_name: str,
        X: Generic[T]
        # grid_search: bool = False
    )-> None:
        if self.model is None:
            self._model_fit(model_name)

        if len(X) > 1:
            raise ValueError(
                "The length of features cannot be greater than 1 for Single Output Prediction task"
            )
        return self.model.predict(X)


class MultiStepOutput(TimeSeriesForecast):
    def __init__(
        self,   
        X: Generic[T],
        y: Generic[T],
        cv_config: CV = ("tss", 5, True)
    ) -> None:
        super().__init__(
            X=X, 
            y=y, 
            cv_config=cv_config
        )
        self.ys = []
        

    def default_split(
        self, 
        train_size: float = 0.8, 
        indices: bool = True
    ) -> List[Generic[T]]:
        return super().default_split(train_size, indices)

    #!
    def forecast(
        self,
        X: Generic[T], #* now works with df only
        slice_window: List[int] | List[str] | str, #* if str given search for columns where str is in
        model_name: str = "GBR",
        
    ):
        '''
        #* The key concept of reverse 
        #* multistep output prediction is
        #* the use of early predicted data as an input
        #* for a next prediction
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
        if self.model is None:
            self._model_fit(model_name)

        
        
        
        for n, i in enumerate(X.index):
            
            yi = self.model.predict([X.loc[i, :]])
            self.ys.append(*yi)
            
            current_window = X.loc[i, slice_window].to_numpy()
            
            moved_window = [*yi, *current_window[:-1]]
            X.loc[i+1, slice_window] = moved_window
            
            if n+1 == len(X):
                break
            

    def _predict(self, step):
        self.model


    def recursive_forecast(self):
        return
