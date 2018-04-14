## plot data og lede efter correlation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import svd
from scipy import stats

plt.style.use('classic') # Set plot theme

data=pd.read_csv('FINALDATA.csv',sep=';')
dates = data['DateTime_Out']
x = [dt.datetime.strptime(d,'%d/%m/%Y %H.%M') for d in dates]

data = [data,x]
################################3
## Summary statistics

data_corr = data.corr() #Returns the correlation between columns in a DataFrame

PD_desc = pimaData.describe() #Summary statistics for numerical columns
PD_desc.to_excel('summaryStat.xlsx')

PD_cov  = pimaData.cov() # Return the covariance between columns in a DataFrame 

PD_var  = pimaData.var()  # Return the variance between columns in a DataFrame 

