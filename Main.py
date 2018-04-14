#Main script
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import svd
from scipy import stats
from datetime import datetime

plt.style.use('classic') # Set plot theme

batches=pd.read_csv('hermikkel.csv',sep = ';')  # Read the data and name alle the attributes
pd.to_datetime(batches['StartTime'], format='%Y-%m-%d %h:%m:%s', errors='ignore')



staffing=pd.read_csv('staffing.csv')

import datetime as dt

dates = staffing['StartTime']
x = [dt.datetime.strptime(d,'%Y-%m-%d %H:%M:%S').date() for d in dates]
y = staffing['Staff']

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d-%Y %h:%m:%s'))
#plt.gca().xaxis.set_major_locator(mdates.DayLocator())
plt.plot(x,y)
plt.gcf().autofmt_xdate()