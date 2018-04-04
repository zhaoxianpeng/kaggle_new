# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     feature
   Description :
   Author :       xpzhao
   date：          18-4-4
-------------------------------------------------
   Change Activity:
                   18-4-4:
-------------------------------------------------
"""
__author__ = 'xpzhao'

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

def show_feature():
    train_data = pd.read_csv('titanic/data/train.csv')
    test_data = pd.read_csv('titanic/data/test.csv')

    sns.set_style('whitegrid')
    print(train_data.head())
    # 数据信息总览
    print(train_data.info())
    print("-" * 40)
    print(test_data.info())

    # Embarked这一属性（共有三个上船地点），缺失俩值，可以用众数赋值
    train_data.Embarked[train_data.Embarked.isnull()] = train_data.Embarked.dropna().mode().values

    #对于标称属性，可以赋一个代表缺失的值，比如‘U0’。因为缺失本身也可能代表着一些隐含信息。
    # 比如船舱号Cabin这一属性，缺失可能代表并没有船舱。
    train_data['Cabin'] = train_data.Cabin.fillna('U0')  # train_data.Cabin[train_data.Cabin.isnull()]='U0'