from abc import ABC, abstractmethod
from typing import List, Tuple, Union
import numpy as np
from sklearn import linear_model


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



class TempsFlowRates(Analytics):
    def __init__(self):
        return


class Currents(Analytics):
    def __init__(self):
        return