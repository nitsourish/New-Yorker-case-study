import pickle
import gc
import os
import sys
from collections import defaultdict
from timeit import default_timer as timer
import numpy as np
import pandas as pd
import xgboost as xgb
from fastai.tabular.core import add_datepart
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    ParameterGrid,
    ParameterSampler,
    RandomizedSearchCV,
    StratifiedKFold,
    StratifiedShuffleSplit,
    cross_val_score,
    train_test_split,
)
from datetime import datetime, timedelta
from sktime.forecasting.compose import TransformedTargetForecaster
from sktime.transformations.series.detrend import Deseasonalizer, Detrender
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose import (
    make_reduction,
    MultioutputTimeSeriesRegressionForecaster,
    DirectTabularRegressionForecaster,
    MultioutputTabularRegressionForecaster,
)
from sktime.forecasting.model_selection import (
    temporal_train_test_split,
    SingleWindowSplitter,
    SlidingWindowSplitter,
)
from sktime.forecasting.model_selection import ForecastingRandomizedSearchCV
from statsmodels.tsa.stattools import adfuller
from termcolor import colored
import logging
from utility import data_prep
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, ClassifierMixin

model_path = "../models/"
path = "../forecast1/"
data_path = "../data/"


class Model_Inference(TransformerMixin, RegressorMixin, BaseEstimator):

    """Class for using model for prediction
    Input: Featurized data at daily level
    """

    def __init__(self, model_path):

        self.model_path = model_path

    def collect_data_model(self, data):
        test_x = data.drop("product_code", axis=1)
        with open(self.model_path, "rb") as fout:
            model = pickle.load(fout)
        return model, test_x

    def make_forecast(self, model, test_x):

        """Method for Forecasting"""

        pred_y = round(model.predict(fh=model.fh, X=test_x), 0)
        return pred_y


prediction = pd.DataFrame()
if __name__ == "__main__":
    test_df = pd.read_csv(os.path.join(data_path, "test_data.csv"))
    with open(os.path.join(model_path, "prod_dict.p"), "rb") as fp:
        prod_dict = pickle.load(fp)
    for i in list(prod_dict.keys()):
        print(i)
        test_x = test_df[test_df.product_code == i]
        prod_mod_path = os.path.join(model_path, f"item_{i}.pkl")
        mi = Model_Inference(model_path=prod_mod_path)
        model, test_x = mi.collect_data_model(test_x)
        pred = mi.make_forecast(model, test_x)
        pred = pd.DataFrame(
            {
                "prod": prod_dict[i],
                "forecast": pred.sales.values,
                "day": list(range(1, 15)),
            }
        )
        if not os.path.exists(path):
            os.makedirs(path)
        pred.to_csv(os.path.join(path, f"item_{i}.csv"), index=False)
        prediction = prediction.append(pred)
    prediction.to_csv(os.path.join(path, "all_product_prediction.csv"), index=False)
