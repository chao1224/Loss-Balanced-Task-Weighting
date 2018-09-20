from __future__ import print_function

import pandas as pd
import numpy as np
import random
import os
import json
import csv
import re


'''
Read the data from all files in input_file_list
And merged into one dataset
'''
def read_merged_data(input_file_list):
    whole_pd = pd.DataFrame()
    for input_file in input_file_list:
        # # TODO: This is for debugging
        # data_pd = pd.read_csv(input_file, nrows=5000)
        data_pd = pd.read_csv(input_file)
        whole_pd = whole_pd.append(data_pd)
    print('Data shape\t{}'.format(whole_pd.shape))
    return whole_pd


'''
Get the fingerprints, with feature_name specified, and label_name specified
'''
def extract_feature_and_label(data_pd,
                              feature_name,
                              task_list):
    X_data = data_pd[feature_name].tolist()
    X_data = map(lambda x: list(x), X_data)
    X_data = np.array(X_data)

    y_data = data_pd[task_list].values.tolist()
    y_data = np.array(y_data)
    y_data = reshape_data_into_2_dim(y_data)

    X_data = X_data.astype(float)
    y_data = y_data.astype(float)

    return X_data, y_data

def extract_feature_and_label_Kaggle(data_pd,
                                     task_list):
    from kaggle_features import merck_descriptors
    X_data = data_pd[merck_descriptors].values.tolist()
    X_data = np.array(X_data)

    y_data = data_pd[task_list].values.tolist()
    y_data = np.array(y_data)
    y_data = reshape_data_into_2_dim(y_data)

    X_data = X_data.astype(float)
    y_data = y_data.astype(float)

    return X_data, y_data


def transform_dataframe2array(dataframe):
    data = np.array(dataframe.values.tolist())
    return reshape_data_into_2_dim(data)


def filter_missing_values(dataframe):
    columns = dataframe.columns
    for c in columns:
        dataframe[c][dataframe[c].notnull()] = 1
    dataframe.fillna(0, inplace=True)
    return dataframe


'''
Reshape vector into 2-dimension matrix.
'''
def reshape_data_into_2_dim(data):
    if data.ndim == 1:
        n = data.shape[0]
        data = data.reshape(n, 1)
    return data
