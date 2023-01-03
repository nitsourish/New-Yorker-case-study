import pickle
import sklearn
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

logger = logging.getLogger()

data_path = "../data/"
path = "../models1/"
min_date = "2018-09-20"
max_date = "2020-09-22"

x_feats = [
    "Year",
    "Week",
    "Day",
    "Dayofyear",
    "price",
    "Is_month_end",
    "Is_month_start",
    "Is_quarter_end",
    "Is_quarter_start",
    "Is_year_end",
    "Is_year_start",
    "Dayofweek_0",
    "Dayofweek_1",
    "Dayofweek_2",
    "Dayofweek_3",
    "Dayofweek_4",
    "Dayofweek_5",
    "Dayofweek_6",
    "Month_1",
    "Month_2",
    "Month_3",
    "Month_4",
    "Month_5",
    "Month_6",
    "Month_7",
    "Month_8",
    "Month_9",
    "Month_10",
    "Month_11",
    "Month_12",
]
target = "sales"


class FeatureProcessor(TransformerMixin, BaseEstimator, RegressorMixin):

    """
    Feature Engineering class. Option to make at any aggregation level- daily/weekly(changing 'D' to 'W') etc.

    Args:
        params:
            X:Product level transaction data

    Returns:
        a) Product level Featurized(date and daily price) data at daily level
    """

    def transform(self, X):
        X = self.add_date_feats(X)
        X = self.binarize_cat(X)
        X = self.ohe_cat(X)
        return X

    def add_date_feats(self, X):

        """Method to extract date features"""

        X = X.rename(columns={"t_dat": "date"})
        X = X.groupby(["product_code", "date"], as_index=False).agg(
            {"price": ["size", "mean"]}
        )
        X.columns = ["product_code", "date", "sales", "price"]
        X["date"] = pd.to_datetime(X["date"])
        X1 = X[["product_code", "date", "price"]]
        X = X.set_index("date")
        X = pd.DataFrame(X.resample("D").sales.sum()).reset_index(level="date")
        X = pd.merge(X, X1, on="date", how="inner")
        date = X["date"]
        X = add_datepart(X, "date")
        X["date"] = date
        return X

    def binarize_cat(self, X):

        """Method for binary categorical features"""

        bin_col = [
            "Is_month_end",
            "Is_month_start",
            "Is_quarter_end",
            "Is_quarter_start",
            "Is_year_end",
            "Is_year_start",
        ]
        bin = sklearn.preprocessing.LabelBinarizer()
        X[bin_col] = bin.fit_transform(X[bin_col])
        return X

    def ohe_cat(self, X):

        """Method for onehot encode of categorical features"""

        cat_col = ["Dayofweek", "Month"]
        ohe = sklearn.preprocessing.OneHotEncoder(sparse=False)
        ohe.fit(X[cat_col])
        x = ohe.transform(X[cat_col])
        x = pd.DataFrame(x)
        x.columns = ohe.get_feature_names_out()
        X = pd.concat([X.drop(cat_col, axis=1).reset_index(drop=True), x], axis=1)
        return X


class Model_train_prediction(TransformerMixin, RegressorMixin, BaseEstimator):

    """Class for building model and prediction including evaluation
    Input: Product level Featurized data at daily level
    """

    def __init__(self, window_len, test_size, path):

        """test_size- Period of Forecasting(2 weeks)
        window_len- number of lags sliding window transformation
        """
        self.test_size = test_size
        self.window_len = window_len
        self.path = path
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def data_prep(self, data):

        """Method to create train and test data"""

        X = data[x_feats]
        Y = data[[target]]
        date = data["date"]
        train_y, test_y, train_x, test_x = temporal_train_test_split(
            y=Y, X=X, test_size=self.test_size
        )
        return train_y, test_y, train_x, test_x

    def model_train(self, train_x, train_y, test_y):

        """Method to train Model using grid search"""

        fh = ForecastingHorizon(values=test_y.index, is_relative=False)
        regressor = xgb.XGBRFRegressor(objective="reg:squarederror", random_state=42)
        forecaster = MultioutputTabularRegressionForecaster(
            estimator=regressor, window_length=self.window_len
        )
        forecaster.fit(y=train_y, X=train_x, fh=fh)
        return forecaster, fh

    def make_forecast(self, forecaster, test_x, fh):

        """Method for Forecasting"""

        pred_y = round(forecaster.predict(X=test_x, fh=fh), 0)
        return pred_y

    def evaluate(self, test_y, pred_y):

        """Method to evaluate Forecasting result"""

        mape = sklearn.metrics.mean_absolute_percentage_error(test_y, pred_y)
        return mape


mape = {}
prediction = {}
actual = {}
if __name__ == "__main__":

    # data prep
    train = pd.read_csv(os.path.join(data_path, "transactions_train.csv.zip"))
    products = pd.read_csv(os.path.join(data_path, "articles.csv.zip"))
    print("Data Read")
    df = data_prep(
        transaction=train, products=products, penetration=0.01, cum_sales_fraction=0.0
    )
    print(f"Data prepared {df.shape}")

    # Consistent products
    duration = df.groupby(["product_code"]).agg({"t_dat": ["min", "max"]})
    duration.columns = ["min_date", "max_date"]
    consistent_prods = list(
        duration[(duration.min_date <= min_date) & (duration.max_date >= max_date)].index
    )
    print(f"count of products for long time {len(consistent_prods)}")
    df = df[df.product_code.isin(consistent_prods)]

    # modeling
    mtp = Model_train_prediction(window_len=30, test_size=14, path=path)
    for i in consistent_prods:
        fp = FeatureProcessor()
        train_feat = fp.transform(df[df.product_code == i])
        train_y, test_y, train_x, test_x = mtp.data_prep(train_feat)
        forecaster, fh = mtp.model_train(train_x, train_y, test_y)
        with open(os.path.join(path, f"item_{i}.pkl"), "wb") as f:
            pickle.dump(forecaster, f)
        pred_y = mtp.make_forecast(forecaster, test_x, fh)
        prediction[i] = list(pred_y.sales.values)
        actual[i] = list(test_y.sales.values)
        try:
            mape[i] = mtp.evaluate(test_y, pred_y)
        except:
            mape[i]: ZeroDivisionError
    actuals = pd.DataFrame(actual)
    actuals.columns = [f"actual_{i}" for i in actual.keys()]
    predictions = pd.DataFrame(prediction)
    predictions.columns = [f"prediction_{i}" for i in prediction.keys()]
    acutal_prediction = pd.concat([actuals, predictions], axis=1)
    acutal_prediction.to_csv(os.path.join(path, "acutal_prediction.csv"), index=False)
    print(acutal_prediction.head())
