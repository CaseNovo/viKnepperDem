## plot data og lede efter correlation
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import svd
from scipy import stats

plt.style.use('classic') # Set plot theme

data=pd.read_csv('FINALDATAR.csv',sep=',')
dates = data['DateTime_Out']
date = [dt.datetime.strptime(d,'%Y-%m-%d %H:%M:%S') for d in dates]
data = data.drop(['DateTime_Out'],axis=1) 
data['date']=date
################################3
## Summary statistics

data_corr = data.corr() #Returns the correlation between columns in a DataFrame

PD_desc = data.describe() #Summary statistics for numerical columns
PD_desc.to_excel('summaryStat.xlsx')

PD_cov  = data.cov() # Return the covariance between columns in a DataFrame 

PD_var  = data.var()  # Return the variance between columns in a DataFrame 

