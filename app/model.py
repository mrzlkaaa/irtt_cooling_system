from __future__ import annotations
from typing import Dict, List, Union
from sklearn.model_selection import cross_validate
import numpy as np
import pandas as pd


class TrainModel:
    def __init__(self, X, y) -> None:
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

    #?
    def fit(self, model):
        return

    #?
    def predict(self, model):
        return
    #?
    def get_score(self):
        return
