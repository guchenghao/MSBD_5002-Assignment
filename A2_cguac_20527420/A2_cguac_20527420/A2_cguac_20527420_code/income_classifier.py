#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /Users/guchenghao/Desktop/DataMing-5002/A2_cguac_20527420/A2_cguac_20527420_code/income_classifier.py
# Project: /Users/guchenghao/Desktop/DataMing-5002/A2_cguac_20527420/A2_cguac_20527420_code
# Created Date: Thursday, October 11th 2018, 4:00:43 pm
# Author: Harold Gu
# -----
# Last Modified: Thursday, 11th October 2018 4:00:44 pm
# Modified By: Harold Gu
# -----
# Copyright (c) 2018 HKUST
# #
# All shall be well and all shall be well and all manner of things shall be well.
# We're doomed!
###
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier
# from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import lightgbm as lgb

train_data = pd.read_csv(
    './A2_cguac_20527420_code/train_feature_new.csv')

test_data = pd.read_csv(
    './A2_cguac_20527420_code/test_feature_new.csv')

label_Y = pd.read_csv('./data/trainLabels.csv', names=['income'])

label_Y = np.array(label_Y).reshape(-1)


# numerical_features = train_data.select_dtypes(exclude=["object"]).columns
# categorical_features = train_data.select_dtypes(include=["object"]).columns


# robSc = RobustScaler()
# train_data.loc[:, numerical_features] = robSc.fit_transform(
#     train_data.loc[:, numerical_features])
# test_data.loc[:, numerical_features] = robSc.fit_transform(
#     test_data.loc[:, numerical_features])


# X_train, X_test, Y_train, Y_test = train_test_split(
#     train_data, label_Y, test_size=0.2, random_state=66)


def get_train_scores(model):  # ! 计算训练集上的得分

    acc = cross_val_score(model, train_data, label_Y, cv=10,
                          scoring='accuracy', n_jobs=4).mean()

    return acc


# def get_test_scores(model):  # ! 计算训练集上的得分

#     acc = cross_val_score(model, X_test, Y_test, cv=10,
#                           scoring='accuracy', n_jobs=2).mean()

#     return acc


# ! 200, 0.2
# ! 360 0.2 4  87.2239
# ! 400 0.1 5 8 0.8726493118168142
# ! 360 0.1 5 8 0.8727372022229106
# ! 400 0.1 4 8 0.8727662109861377
gbdt_model = GradientBoostingClassifier(n_estimators=400, learning_rate=0.1, max_depth=4, max_features=8)

# Rf_model = RandomForestClassifier(n_estimators=100, max_depth=8)

gbdt_model = gbdt_model.fit(train_data, label_Y)
# Rf_model = Rf_model.fit(train_data, label_Y)

acc = gbdt_model.score(train_data, label_Y)
# acc = Rf_model.score(train_data, label_Y)

predictions = gbdt_model.predict(test_data)
# predictions = Rf_model.predict(test_data)

submission_csv = pd.Series(predictions)

submission_csv.to_csv('./A2_cguac_25227420_prediction.csv', index=False)

print("Accuracy of 测试集: {0}".format(acc))

print(get_train_scores(gbdt_model))
# print(get_train_scores(Rf_model))
# print(get_test_scores(gbdt_model))
