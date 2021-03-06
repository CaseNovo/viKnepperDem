#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 13:58:53 2018

@author: ibenfjordkjaersgaard
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import svd
from scipy import stats
from datetime import datetime

plt.style.use('classic') # Set plot theme



staffdk=pd.read_csv('staffdk.csv')

import datetime as dt
staff1 = staffdk[staffdk['Line']==1]
staff2 = staffdk[staffdk['Line']==2]

staff1['StartTime_datetime'] = pd.to_datetime(staff1['StartTime'])
staff1['EndTime_datetime'] = pd.to_datetime(staff1['EndTime'])
#staff1 = staff1.drop('Line',axis= 1)
#staff1 = staff1.set_index('StartTime_datetime')



df1=staff1.copy()
df2=staff1.copy()
df1.index=df1['StartTime_datetime']
df2.index=df2['EndTime_datetime']
df_final=df1.groupby(pd.TimeGrouper('h')).mean()['Staff'].fillna(0).subtract(df2.groupby(pd.TimeGrouper('h')).mean()['Staff'].fillna(0),fill_value=0).cumsum()

data1=df_final
np.savetxt('datastaff1.csv',data1,delimiter=',')


staff2['StartTime_datetime'] = pd.to_datetime(staff2['StartTime'])
staff2['EndTime_datetime'] = pd.to_datetime(staff2['EndTime'])

df1=staff2.copy()
df2=staff2.copy()
df1.index=df1['StartTime_datetime']
df2.index=df2['EndTime_datetime']
df_final=df1.groupby(pd.TimeGrouper('h')).mean()['Staff'].fillna(0).subtract(df2.groupby(pd.TimeGrouper('h')).mean()['Staff'].fillna(0),fill_value=0).cumsum()

data2=df_final
np.savetxt('datastaff2.csv',data2,delimiter=',')
