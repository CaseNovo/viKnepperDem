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

batches1 =