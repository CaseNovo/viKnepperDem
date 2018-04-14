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
from datetime import datetime
import datetime as dt
plt.style.use('classic') # Set plot theme

stopdk=pd.read_csv('line12.csv',sep=';')
sum(stopdk.Error)


#staff1['StartTime_datetime'] = pd.to_datetime(staff1['StartTime'])
#staff1['EndTime_datetime'] = pd.to_datetime(staff1['EndTime'])
#staff1 = staff1.drop('Line',axis= 1)
#staff1 = staff1.set_index('StartTime_datetime')


df1=stopdk.Error.copy()
df2=stopdk.Error.copy()
df1.index=stopdk.StartTime_datetime

df2.index=stopdk['EndTime_dt']
df_final=df1.groupby(pd.TimeGrouper('h')).mean()['Staff'].fillna(0).subtract(df2.groupby(pd.TimeGrouper('h')).mean()['Staff'].fillna(0),fill_value=0).cumsum()


dates = stopdk['StartTime']
x = [dt.datetime.strptime(d,'%d-%m-%y %H:%M').date() for d in dates]
y=stopdk.Error

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d-%Y %h:%m:%s'))
#plt.gca().xaxis.set_major_locator(mdates.DayLocator())
plt.plot(x,y)
plt.gcf().autofmt_xdate()