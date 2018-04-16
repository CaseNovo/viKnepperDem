import pandas as pd
import numpy as np
data=pd.read_csv('StopMikkel1.csv',sep=';')
DFpenn = pd.read_csv('DFpennr.csv',sep='\t')
df = data
D = pd.DataFrame(columns=['signal', 'duration'])
#df = df.drop(['Signal']==['Machine stopped manually'],axis=0)
sum(DFpenn['OutputGood'])

starving = df[df.Signal=='Starving']
np.mean(starving['Duration'])

D['signal'] =  df['Signal'].unique()
errortype=[None]*len(D['signal'])

errortype = df['Signal'].unique()
errorDuration =[None]*len(errortype)
for i in range(len(errortype)):
    DU = df[df['Signal']==errortype[i]]
    errorDuration[i]= sum(DU['Duration'])

errortype= list(errortype)
#errortype['duration'] =errorDuration
D['duration'] =errorDuration




#D.sort_values(by=['duration'],ascending=False)
D = D.drop(['signal']==['Running'],axis=0)
totalDown =sum(D['duration'])
#D = D.drop(['signal']==['Unknown State'],axis=0)
#D = D.drop(['signal']==['Machine stopped manually'],axis=0)
D.to_csv('D.csv', sep='\t')
starving.to_csv('starvingStop.csv',sep='\t')
allStarving=sum(starving['Duration'])
meanStarving = np.mean(starving['Duration'])
starving = starving[starving.Duration>10]
bigStarving=sum(starving['Duration'])



bigStarving/allStarving # 1/3 af tiden starver vi mere end 10 sek

bigFejlMin =bigStarving - len(starving['Duration'])*5 # Hvordan ser det ud hvis vi min fejl


costStarvingOpti = bigFejlMin/60*188*2*0.75 # sænkes fejl til mean = 5 sek kan vi der tjenes 442.000 kroner per 3 måneder.




# Finder Starving DK
starvingDK = starving[starving['Line']!=3] 
starvingDK = starvingDK[starvingDK['Line']!=4] 
bigStarvingDK=sum(starving['Duration'])

bigStarvingDK/bigFejlMin # 1/3 af tiden starver vi mere end 10 sek

bigFejlMinDK =bigStarvingDK - len(starvingDK['Duration'])*5 # Hvordan ser det ud hvis vi min fejl


costStarvingOptiDK = bigFejlMinDK/60*188*2 
bigStarvingDK=sum(starvingDK['Duration'])

bigStarvingDK/bigFejlMin 

bigFejlMinDK =bigStarvingDK - len(starvingDK['Duration'])*5 # Hvordan ser det ud hvis vi min fejl


costStarvingOptiDK = bigFejlMinDK/60*188*2 *0.75# sænkes fejl til mean = 5 sek kan vi der tjenes 357.500 kroner per 3 måneder.



# Finder Starving sumba
starvingBrazil = starving[starving['Line']!=1] 
starvingBrazil = starvingBrazil[starvingBrazil['Line']!=2] 
bigStarvingBrazil=sum(starving['Duration'])

bigStarvingBrazil/bigFejlMin # 1/3 af tiden starver vi mere end 10 sek

bigFejlMinBrazil =bigStarvingBrazil - len(starvingBrazil['Duration'])*5 # Hvordan ser det ud hvis vi min fejl


costStarvingOptiBrazil = bigFejlMinBrazil/60*188*2 *0.75
bigStarvingBrazil=sum(starvingBrazil['Duration'])

bigStarvingBrazil/bigFejlMin 

bigFejlMinBrazil =bigStarvingBrazil - len(starvingBrazil['Duration'])*5 # Hvordan ser det ud hvis vi min fejl


costStarvingOptiBrazil = bigFejlMinBrazil/60*188*2 *0.75# sænkes fejl til mean = 5 sek kan vi der tjenes 357.500 kroner per 3 måneder.


costStarvingOptiDK = costStarvingOpti-costStarvingOptiBrazil



# hvem har størst starv % 
prodDK = DFpenn[DFpenn['Line']!=3]
prodDK = prodDK[prodDK['Line']!=4]
totalProdDK = sum(prodDK['OutputGood'])

prodBrazil = DFpenn[DFpenn['Line']!=1]
prodDK = prodBrazil[prodBrazil['Line']!=2]
totalProdBrazil = sum(prodBrazil['OutputGood'])

totalProdBrazil = totalProdBrazil/(totalProdBrazil+totalProdDK)
## Danmark producere 2 gange mere og har laveste starv


# Hvis brazil får lige så god starving som DK
gevinst=bigStarvingBrazil - bigStarvingDK*1/totalProdBrazil
gevingcost = gevinst/60*188*2*0.75



prodDK1 = DFpenn[DFpenn['Line']==1]
prodDK1 = sum(prodDK1['OutputGood'])

prodDK2 = DFpenn[DFpenn['Line']==2]
prodDK2 = sum(prodDK2['OutputGood'])

prodDK1/prodDK2


prodBrazil1 = DFpenn[DFpenn['Line']==3]
prodBrazil1 = sum(prodBrazil1['OutputGood'])

prodBrazil2 = DFpenn[DFpenn['Line']==4]
prodBrazil2 = sum(prodBrazil2['OutputGood'])

prodBrazil1/prodBrazil2
