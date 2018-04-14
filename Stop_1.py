# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 13:12:36 2018

@author: Emilie
"""
#Main script
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import svd
from scipy import stats
from datetime import *

plt.style.use('classic') # Set plot theme

stopdk=pd.read_csv('line12.csv',sep=';')
sum(stopdk.Error)



import datetime as dt
Line1 = stopdk[stopdk['Line']==1]
Line2 = stopdk[stopdk['Line']==2]

Line1['StartTime_datetime'] = pd.to_datetime(Line1['StartTime'])
Line1['EndTime_datetime'] = pd.to_datetime(Line1['EndTime'])
Line1 = Line1.drop('Line',axis= 1)
#staff1 = staff1.set_index('StartTime_datetime')

ss=stopdk.groupby('StartTime')

"""
df1=Line1.copy()
df2=Line1.copy()
df1.index=df1['StartTime_datetime']
df2.index=df2['EndTime_datetime']
df_final=df1.groupby(pd.TimeGrouper('h')).mean()['Error'].fillna(0).subtract(df2.groupby(pd.TimeGrouper('h')).mean()['Error'].fillna(0),fill_value=0).cumsum()
"""