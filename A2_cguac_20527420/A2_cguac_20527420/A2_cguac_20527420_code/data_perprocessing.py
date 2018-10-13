#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /Users/guchenghao/Desktop/DataMing-5002/A2_cguac_20527420/A2_cguac_20527420_code/data_perprocessing.py
# Project: /Users/guchenghao/Desktop/DataMing-5002/A2_cguac_20527420/A2_cguac_20527420_code
# Created Date: Wednesday, October 10th 2018, 3:11:29 pm
# Author: Harold Gu
# -----
# Last Modified: Thursday, 11th October 2018 4:00:22 pm
# Modified By: Harold Gu
# -----
# Copyright (c) 2018 HKUST
# #
# All shall be well and all shall be well and all manner of things shall be well.
# We're doomed!
###


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as son
from scipy.stats import skew
from scipy.stats import norm
from sklearn.ensemble import GradientBoostingClassifier


train_data = pd.read_csv('./data/trainFeatures.csv', header=None, names=[
                         'age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country'])

train_data.drop([0], inplace=True)

test_data = pd.read_csv(
    './data/testFeatures.csv', header=None, names=[
        'age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country'])

test_data.drop([0], inplace=True)


label_Y = pd.read_csv('./data/trainLabels.csv', names=['income'])


train_data[['age', 'fnlwgt', 'capital_loss', 'capital_gain', 'hours_per_week']] = train_data[['age', 'fnlwgt', 'capital_loss', 'capital_gain', 'hours_per_week']].astype(np.int64)

test_data[['age', 'fnlwgt', 'capital_loss', 'capital_gain', 'hours_per_week']] = test_data[['age', 'fnlwgt', 'capital_loss', 'capital_gain', 'hours_per_week']].astype(np.int64)

print('训练数据集的维度: {0}'.format(train_data.shape))
print('测试数据集的维度: {0}'.format(test_data.shape))


print('训练数据的描述统计量: \n {0}'.format(train_data.describe()))


print('训练数据的属性信息: \n {0}'.format(train_data.info()))


# ! 划分数值型特征与类别型特征
numerical_features = train_data.select_dtypes(exclude=["object"]).columns
categorical_features = train_data.select_dtypes(include=["object"]).columns

print('数值类型的特征: \n {0}'.format(numerical_features))
print('类别类型的特征: \n {0}'.format(categorical_features))


train_notNull_features = []
test_notNull_features = []
train_hasNull_features = []
test_hasNull_features = []

for item in list(train_data.columns):
    if ' ?' in list(train_data[item]):
        train_hasNull_features.append(item)
        # ! train_data[item][train_data[item] == ' ?'] = None
        train_data[item] = train_data[item].replace({' ?': np.nan})
        # train_data[item].fillna(train_data[item].mean(), inplace=True)
    else:
        train_notNull_features.append(item)


for item in list(test_data.columns):
    if ' ?' in list(test_data[item]):
        test_hasNull_features.append(item)
        test_data[item] = test_data[item].replace({' ?': np.nan})
    else:
        test_notNull_features.append(item)

print("train无缺失值的属性列表: \n {0}".format(train_notNull_features))
print("test无缺失值的属性列表: \n {0}".format(test_notNull_features))
print('\n')
print("train有缺失值的属性列表: \n {0}".format(train_hasNull_features))
print("test有缺失值的属性列表: \n {0}".format(train_hasNull_features))

# ! 画图分析
plot_train_data = pd.concat([train_data, label_Y], axis=1)
son.set(style="ticks", color_codes=True)
# son.relplot(x='hours_per_week', y='capital_gain',
#             data=plot_train_data, hue='income')
# ! 计数图
son.distplot(plot_train_data['fnlwgt'], kde=True, bins=50)
son.catplot(y='education', hue='income', palette="pastel", edgecolor=".6", data=plot_train_data, kind='count')
son.catplot(y='workclass', hue='income', palette="pastel", edgecolor=".6", data=plot_train_data, kind='count')

plt.show()


skewness = train_data[numerical_features].apply(lambda x: skew(x))
print(skewness)
skewness = skewness[abs(skewness) > 0.5]  # ! 偏态值大于0.5，全部是正偏态
print(str(skewness.shape[0]) + " skewed numerical features to log transform")
skewed_features = skewness.index
# ! 将被定为正偏态的值进行数据转换
train_data[skewed_features] = np.log1p(train_data[skewed_features])

skewness = test_data[numerical_features].apply(lambda x: skew(x))
print(skewness)
skewness = skewness[abs(skewness) > 0.5]  # ! 偏态值大于0.5，全部是正偏态
print(str(skewness.shape[0]) + " skewed numerical features to log transform")
skewed_features = skewness.index
# ! 将被定为正偏态的值进行数据转换
test_data[skewed_features] = np.log1p(test_data[skewed_features])

train_data['workclass'] = train_data['workclass'].replace({
    ' Private': 0,
    ' Self-emp-not-inc': 1,
    ' Self-emp-inc': 2,
    ' Federal-gov': 3,
    ' Local-gov': 4,
    ' State-gov': 5,
    ' Without-pay': 6,
    ' Never-worked': 7
})

test_data['workclass'] = test_data['workclass'].replace({
    ' Private': 0,
    ' Self-emp-not-inc': 1,
    ' Self-emp-inc': 2,
    ' Federal-gov': 3,
    ' Local-gov': 4,
    ' State-gov': 5,
    ' Without-pay': 6,
    ' Never-worked': 7
})

train_data['sex'] = train_data['sex'].replace({
    ' Male': 0,
    ' Female': 1
})

test_data['sex'] = test_data['sex'].replace({
    ' Male': 0,
    ' Female': 1
})

train_data['education'] = train_data['education'].replace({
    ' Preschool': 0,
    ' 1st-4th': 1,
    ' 5th-6th': 2,
    ' 7th-8th': 3,
    ' 9th': 4,
    ' 10th': 5,
    ' 11th': 6,
    ' 12th': 7,
    ' HS-grad': 8,
    ' Prof-school': 9,
    ' Assoc-acdm': 10,
    ' Assoc-voc': 11,
    ' Some-college': 12,
    ' Bachelors': 13,
    ' Masters': 14,
    ' Doctorate': 15
})


test_data['education'] = test_data['education'].replace({
    ' Preschool': 0,
    ' 1st-4th': 1,
    ' 5th-6th': 2,
    ' 7th-8th': 3,
    ' 9th': 4,
    ' 10th': 5,
    ' 11th': 6,
    ' 12th': 7,
    ' HS-grad': 8,
    ' Prof-school': 9,
    ' Assoc-acdm': 10,
    ' Assoc-voc': 11,
    ' Some-college': 12,
    ' Bachelors': 13,
    ' Masters': 14,
    ' Doctorate': 15
})

train_data['marital_status'] = train_data['marital_status'].replace({
    ' Divorced': 0,
    ' Married-AF-spouse': 1,
    ' Married-civ-spouse': 2,
    ' Married-spouse-absent': 3,
    ' Never-married': 4,
    ' Separated': 5,
    ' Widowed': 6
})

test_data['marital_status'] = test_data['marital_status'].replace({
    ' Divorced': 0,
    ' Married-AF-spouse': 1,
    ' Married-civ-spouse': 2,
    ' Married-spouse-absent': 3,
    ' Never-married': 4,
    ' Separated': 5,
    ' Widowed': 6
})

train_data['occupation'] = train_data['occupation'].replace({
    ' Tech-support': 0,
    ' Craft-repair': 1,
    ' Other-service': 2,
    ' Sales': 3,
    ' Exec-managerial': 4,
    ' Prof-specialty': 5,
    ' Handlers-cleaners': 6,
    ' Machine-op-inspct': 7,
    ' Adm-clerical': 8,
    ' Farming-fishing': 9,
    ' Transport-moving': 10,
    ' Priv-house-serv': 11,
    ' Protective-serv': 12,
    ' Armed-Forces': 13
})

test_data['occupation'] = test_data['occupation'].replace({
    ' Tech-support': 0,
    ' Craft-repair': 1,
    ' Other-service': 2,
    ' Sales': 3,
    ' Exec-managerial': 4,
    ' Prof-specialty': 5,
    ' Handlers-cleaners': 6,
    ' Machine-op-inspct': 7,
    ' Adm-clerical': 8,
    ' Farming-fishing': 9,
    ' Transport-moving': 10,
    ' Priv-house-serv': 11,
    ' Protective-serv': 12,
    ' Armed-Forces': 13
})


train_data['relationship'] = train_data['relationship'].replace({
    ' Wife': 0,
    ' Own-child': 1,
    ' Husband': 2,
    ' Not-in-family': 3,
    ' Other-relative': 4,
    ' Unmarried': 5
})

test_data['relationship'] = test_data['relationship'].replace({
    ' Wife': 0,
    ' Own-child': 1,
    ' Husband': 2,
    ' Not-in-family': 3,
    ' Other-relative': 4,
    ' Unmarried': 5
})

train_data['race'] = train_data['race'].replace({
    ' White': 0,
    ' Asian-Pac-Islander': 1,
    ' Amer-Indian-Eskimo': 2,
    ' Other': 3,
    ' Black': 4
})

test_data['race'] = test_data['race'].replace({
    ' White': 0,
    ' Asian-Pac-Islander': 1,
    ' Amer-Indian-Eskimo': 2,
    ' Other': 3,
    ' Black': 4
})

train_data['native_country'] = train_data['native_country'].replace({
    ' United-States': 0,
    ' Cambodia': 1,
    ' England': 2,
    ' Canada': 3,
    ' Germany': 4,
    ' Outlying-US(Guam-USVI-etc)': 5,
    ' India': 6,
    ' Japan': 7,
    ' Greece': 8,
    ' South': 9,
    ' China': 10,
    ' Cuba': 11,
    ' Iran': 12,
    ' Honduras': 13,
    ' Philippines': 14,
    ' Italy': 15,
    ' Poland': 16,
    ' Jamaica': 17,
    ' Vietnam': 18,
    ' Mexico': 19,
    ' Portugal': 20,
    ' Ireland': 21,
    ' France': 22,
    ' Dominican-Republic': 23,
    ' Laos': 24,
    ' Ecuador': 25,
    ' Taiwan': 26,
    ' Haiti': 27,
    ' Columbia': 28,
    ' Hungary': 29,
    ' Guatemala': 30,
    ' Nicaragua': 31,
    ' Scotland': 32,
    ' Thailand': 33,
    ' Yugoslavia': 34,
    ' El-Salvador': 35,
    ' Trinadad&Tobago': 36,
    ' Peru': 37,
    ' Hong': 38,
    ' Holand-Netherlands': 39,
    ' Puerto-Rico': 40
})


test_data['native_country'] = test_data['native_country'].replace({
    ' United-States': 0,
    ' Cambodia': 1,
    ' England': 2,
    ' Canada': 3,
    ' Germany': 4,
    ' Outlying-US(Guam-USVI-etc)': 5,
    ' India': 6,
    ' Japan': 7,
    ' Greece': 8,
    ' South': 9,
    ' China': 10,
    ' Cuba': 11,
    ' Iran': 12,
    ' Honduras': 13,
    ' Philippines': 14,
    ' Italy': 15,
    ' Poland': 16,
    ' Jamaica': 17,
    ' Vietnam': 18,
    ' Mexico': 19,
    ' Portugal': 20,
    ' Ireland': 21,
    ' France': 22,
    ' Dominican-Republic': 23,
    ' Laos': 24,
    ' Ecuador': 25,
    ' Taiwan': 26,
    ' Haiti': 27,
    ' Columbia': 28,
    ' Hungary': 29,
    ' Guatemala': 30,
    ' Nicaragua': 31,
    ' Scotland': 32,
    ' Thailand': 33,
    ' Yugoslavia': 34,
    ' El-Salvador': 35,
    ' Trinadad&Tobago': 36,
    ' Peru': 37,
    ' Hong': 38,
    ' Holand-Netherlands': 39,
    ' Puerto-Rico': 40
})


# train_data = pd.get_dummies(train_data)
print(train_data['workclass'])
isnulldata = train_data[pd.isnull(train_data['workclass'])]
notnulldata = train_data[pd.notnull(train_data['workclass'])]
workclass_train_classifier = GradientBoostingClassifier()
workclass_train_classifier.fit(notnulldata[train_notNull_features].values, notnulldata['workclass'].values)
predictions = workclass_train_classifier.predict(isnulldata[train_notNull_features].values)
train_data.workclass[pd.isnull(train_data['workclass'])] = predictions
print(train_data['workclass'])

print(test_data['workclass'])
isnulldata = test_data[pd.isnull(test_data['workclass'])]
notnulldata = test_data[pd.notnull(test_data['workclass'])]
workclass_test_classifier = GradientBoostingClassifier()
workclass_test_classifier.fit(
    notnulldata[test_notNull_features].values, notnulldata['workclass'].values)
predictions = workclass_test_classifier.predict(
    isnulldata[test_notNull_features].values)
test_data.workclass[pd.isnull(test_data['workclass'])] = predictions
print(test_data['workclass'])


print(train_data['occupation'])
isnulldata = train_data[pd.isnull(train_data['occupation'])]
notnulldata = train_data[pd.notnull(train_data['occupation'])]
occupation_train_classifier = GradientBoostingClassifier()
occupation_train_classifier.fit(notnulldata[train_notNull_features].values, notnulldata['occupation'].values)
predictions = occupation_train_classifier.predict(isnulldata[train_notNull_features].values)
train_data.occupation[pd.isnull(train_data['occupation'])] = predictions
print(train_data['occupation'])

print(test_data['occupation'])
isnulldata = test_data[pd.isnull(test_data['occupation'])]
notnulldata = test_data[pd.notnull(test_data['occupation'])]
occupation_test_classifier = GradientBoostingClassifier()
occupation_test_classifier.fit(
    notnulldata[test_notNull_features].values, notnulldata['occupation'].values)
predictions = occupation_test_classifier.predict(
    isnulldata[test_notNull_features].values)
test_data.occupation[pd.isnull(test_data['occupation'])] = predictions
print(test_data['occupation'])

print(train_data['native_country'])
isnulldata = train_data[pd.isnull(train_data['native_country'])]
notnulldata = train_data[pd.notnull(train_data['native_country'])]
native_country_train_classifier = GradientBoostingClassifier()
native_country_train_classifier.fit(notnulldata[train_notNull_features].values, notnulldata['native_country'].values)
predictions = native_country_train_classifier.predict(isnulldata[train_notNull_features].values)
train_data.native_country[pd.isnull(train_data['native_country'])] = predictions
print(train_data['native_country'])

print(test_data['native_country'])
isnulldata = test_data[pd.isnull(test_data['native_country'])]
notnulldata = test_data[pd.notnull(test_data['native_country'])]
native_country_test_classifier = GradientBoostingClassifier()
native_country_test_classifier.fit(
    notnulldata[test_notNull_features].values, notnulldata['native_country'].values)
predictions = native_country_test_classifier.predict(
    isnulldata[test_notNull_features].values)
test_data.native_country[pd.isnull(
    test_data['native_country'])] = predictions
print(test_data['native_country'])


print('数据清洗后的数据: {0}'.format(train_data))


train_data.to_csv(
    './A2_cguac_20527420_code/train_feature_new.csv', index=False)


test_data.to_csv(
    './A2_cguac_20527420_code/test_feature_new.csv', index=False)
