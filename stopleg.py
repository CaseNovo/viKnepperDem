import pandas as pd
import numpy as np
data=pd.read_csv('StopMikkel1.csv',sep=';')
df = data
D = pd.DataFrame(columns=['signal', 'duration'])



starving = df[df.Signal=='Starving']
np.mean(starving['Duration'])

D['signal'] =  df['Signal'].unique()


#errortype = df['Signal'].unique()
errorDuration =[None]*len(errortype)
for i in range(len(errortype)):
    DU = df[df['Signal']==errortype[i]]
    errorDuration[i]= sum(DU['Duration'])

errortype= list(errortype)
#errortype['duration'] =errorDuration
D['duration'] =errorDuration




D.sort_values(by=['duration'],ascending=False)
D = D.drop(['signal']==['Running'],axis=0)
totalDown =sum(D['duration'])
D = D.drop(['signal']==['Unknown State'],axis=0)
D = D.drop(['signal']==['Machine stopped manually'],axis=0)
D.to_csv('D.csv', sep='\t')
starving.to_csv('starvingStop.csv',sep='\t')


unknownState = df[df.signal=='Unknown State']