#Main script
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

dates = staff1['StartTime']
x = [dt.datetime.strptime(d,'%Y-%m-%d %H:%M:%S').date() for d in dates]
y = staff1['Staff']

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d-%Y %h:%m:%s'))
#plt.gca().xaxis.set_major_locator(mdates.DayLocator())
plt.plot(x,y)
plt.gcf().autofmt_xdate()