import copy
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from typing import Callable
from abc import ABCMeta, abstractmethod

## no inheritance for less obscurity METACLEANFCS
def drop_outliers(
    data: pd.DataFrame,
    outlier_func: Callable,
    drop_and: bool = True
):# -> np.ndarray:
    """Applies outlier removal function to given data.

    Args:
        data (pd.DataFrame): Data from which outlier are detected.
        outlier_func (Callable): Outlier detection function.
        drop_and (bool, optional): If True, sets an object as not an outlier if
            all (AND) and its feature values were not flagged as outliers.
            Otherwise, it is not an outlier if any (OR) or its feature values
            were not flagged as outliers.

    Returns:
        np.ndarray: A boolean 1D array the same size as the number of rows in
            `data` labelling non-outliers as True.
    """
    out_keep = ok = np.full((len(data)), 0)
    data_ = copy.copy(data)
    data_['bin'] = np.array(range(len(data))) + 1 # TODO: legacy code
    for c in data.columns:
        du = data_[['bin', c]].drop_duplicates(
            ignore_index=True).copy(deep=True)
        # note: unscaled, works fine
        train_data = np.array(du[c]).reshape(-1, 1)
        train_data[np.isinf(train_data)] = 0
        train_data[np.isnan(train_data)] = 0
        out, _ = outlier_func.fit_predict(train_data, score=True)
        out_keep = out_keep + np.abs(np.minimum(out, ok))

    out_keep = out_keep == 0 if drop_and else out_keep < len(data.columns)
    return out_keep

class OutlierDetector(metaclass=ABCMeta):
    """Super class for `IsoForestDetector` class.
    """
    def __init__(self,  model, **kwargs):
        self.model = model(**kwargs)

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def fit_predict(self):
        pass

class IsoForestDetector(OutlierDetector):
    """Wrapper for outlier detection model, `sklearn.ensemble.IsolationForest`.
    """
    def __init__(
        self,
        model = IsolationForest,
        contamination: float = 0.01,
        n_estimators: int = 500,
        random_state: int = 123
    ):# -> None:
        setting = {
            "contamination":contamination,
            "n_estimators": n_estimators,
            "random_state": random_state
        }
        super(IsoForestDetector, self).__init__(model, **setting)
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, data):
        self.model.fit(data)

    def predict(self, data: np.ndarray, score: bool = True):
        pred = self.model.predict(data)
        if score:
            score_ = self.model.decision_function(data)
        return pred, score_

    def fit_predict(self, data: np.ndarray, score: bool = True):
        self.fit(data)
        return self.predict(data, score=score)

