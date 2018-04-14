## plot data og lede efter correlation
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import svd
from scipy import stats

plt.style.use('classic') # Set plot theme

data=pd.read_csv('FINALDATA3.csv',sep=',')
dataSek = data
dates = data['DateTime_Out']
date = [dt.datetime.strptime(d,'%Y-%m-%d %H:%M:%S') for d in dates]
data = data.drop(['DateTime_Out'],axis=1) 
data = data.drop(['ProductDescription'],axis=1) 
#data['date']=date


dateSek = dataSek['DateTime_Out']
dateSek = [dt.datetime.strptime(d,'%Y-%m-%d %H:%M:%S').timestamp() for d in dates]
dataSek = dataSek.drop(['DateTime_Out'],axis=1) 
dataSek['date']=dateSek

################################3
## Summary statistics

data_corr = data.corr() #Returns the correlation between columns in a DataFrame

PD_desc = data.describe() #Summary statistics for numerical columns
PD_desc.to_excel('summaryStat.xlsx')

PD_cov  = data.cov() # Return the covariance between columns in a DataFrame 

PD_var  = data.var()  # Return the variance between columns in a DataFrame 

names = list(data) # list the names of the attributtes








X = data.values


X = stats.zscore(X)


a = 0
plt. figure()

for i in range(len(names)-1):

    for j in range(len(names)-1):
        a += 1
        plt.subplot(6,6,a)
        plt.scatter(X[:,i],X[:,j], s=1, c='black') # Make scatterplot of i'th and j'th attribute

        plt.xticks([]) # Remove x axes
        plt.yticks([]) # Remove y axes
        n = str(i)
plt.savefig('correlation',dpi=1000)

plt.show()

