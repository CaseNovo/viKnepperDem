import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import svd
from scipy import stats
from datetime import datetime

plt.style.use('classic') # Set plot theme



staffdk=pd.read_csv('datastaff.csv')
batches=pd.read_csv('batches.csv',sep=',')



import datetime as dt
staff1 = batches[batches['Line']==1]
staff2 = batches[batches['Line']==2]

staff1['StartTime_datetime'] = pd.to_datetime(staff1['StartTime'])
staff1['EndTime_datetime'] = pd.to_datetime(staff1['EndTime'])
#staff1 = staff1.drop('Line',axis= 1)
#staff1 = staff1.set_index('StartTime_datetime')


staff1.set_index('StartTime_datetime').resample('M')


df1=staff1.copy()
df2=staff1.copy()
df1.index=df1['StartTime_datetime']
df2.index=df2['EndTime_datetime']
df_final=df1.groupby([pd.TimeGrouper('h'),'ProductDescription'])
df_final=df1.groupby([pd.TimeGrouper('h'),'ProductDescription']).fillna(0).subtract(df2.groupby([pd.TimeGrouper('h'),'ProductDescription']).fillna(0),fill_value=0).cumsum()