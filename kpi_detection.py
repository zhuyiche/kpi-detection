import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from sklearn import preprocessing
from sklearn.preprocessing import scale, minmax_scale, maxabs_scale
import random

# windows
# these can be used as different feature for deep forest
W = np.asarray([2, 7, 12, 20, 30 ,40, 50, 60, 75, 120])
delay = 15

def show_dataset_info(df):
    print(df.columns, '\n')
    print(df.describe(), '\n')
    print(df.head(), '\n')

"""
these were written by xxx and used for public kpi
"""
phase1_train = pd.read_csv('', nrows=10000)
phase1_test = pd.read_csv('').ix[1:10000, :]

show_dataset_info(phase1_train)
show_dataset_info(phase1_test)


def get_time_series_from_dataframe(df):
    """

    :param df:
    :return:
    """
    ts_ids, ts_indexes, ts_point_counts = np.unique(df['KPI ID'],
                                                    return_index=True,
                                                    return_counts=True)
    print('Extract are %d time series in the dataframe:' % (len(ts_ids)))

    ts_indexes.sort()
    ts_indexes = np.append(ts_indexes, len(df))

    set_of_time_series = []
    set_of_time_series_label = []

    for i in np.arange(len(ts_indexes) - 1):
        print('Extracting %d th time series with index %d and %d (exclusive)'
              % (i, ts_indexes[i], ts_indexes[i+1]))
        set_of_time_series.append(np.asarray(df['value'][ts_indexes[i]:ts_indexes[i+1]]))
        set_of_time_series_label.append(np.asarray(df['label'][ts_indexes[i]:ts_indexes[i+1]]))


    return set_of_time_series, set_of_time_series_label



train_time_series



from statsmodels.tsa.api import SARIMAX, ExpoentialSmoothing, SimpleExpSmoothing, Holt

def get_feature_logs(time_series):
    return np.log(time_series + 1e-2)

def get_feature_SARIMA_residuals(time_series):
    predict = SARIMAX(time_series, trend='n', order=(5,1,1), measurement_error=True,
                      enforce_stationarity=False, enforce_invertibility=False).fit().get_prediction()

    return time_series - predict.predicted_mean

def get_feature_AddES_residuals(time_series):
    predict = ExpoentialSmoothing(time_series, trend='add').fit(smoothing_level=1)
    return time_series - predict.fittedvalues

def get_feature_SimplesES_residuals(time_series):
    predict = SimpleExpSmoothing(time_series).fit(smoothing_level=1)
    return time_series - predict.fittedvalues

def get_feature_Holt_residual(time_series):
    predict = Holt(time_series).fit(smoothing_level=1)
    return time_series - predict.fittedvalues

def get_features_and_labels_from_a_time_series(time_series, time_series_label, Windows, delay):
    """

    :param time_series:
    :param time_series_label:
    :param Windows:
    :param delay:
    :return:
    """
    data = []
    data_label = []
    data_label_vital = []

    start_point = 2 * max(Windows)
    start_accum = 0

    time_series_SARIMA_residuals = get_feature_SARIMA_residuals(time_series)
    time_series_AddES_residuals = get_feature_AddES_residuals(time_series)
    time_series_SimpleES_residuals = get_feature_SimplesES_residuals(time_series)
    time_series_Holt_residuals = get_feature_Holt_residual(time_series)

    time_series_logs = get_feature_logs(time_series)




























