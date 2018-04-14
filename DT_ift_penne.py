# exercise 6.1.2

from matplotlib.pyplot import figure, plot, xlabel, ylabel, legend, show, boxplot, title
from sklearn import model_selection, tree
import numpy as np
import pandas as pd
from scipy import stats
from scipy.io import loadmat
# exercise 6.2.1

import sys
sys.path.append("C:/Users/Emilie/Dropbox/Skole/DTU/4. semester/Machine learning og datamining/02450Toolbox_Python/Tools")
from matplotlib.pyplot import figure, plot, subplot, title, xlabel, ylabel, show, clim
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import model_selection
from toolbox_02450 import feature_selector_lr, bmplot
import numpy as np
import pandas as pd
from scipy import stats

# Load data from matlab file
DF=pd.read_csv('FINALDATA3.csv',header=0,sep=',')
DF.replace([np.inf, -np.inf], np.nan)
DF.replace([np.inf, -np.inf], np.nan).dropna()



DF.ProductDescription.replace(['Liraglutid 1.8 mg', 'NovoMix 30 GLY (CCH)','Liraglutid 0.9 mg',
                               'NovoMix 50 NN2000','PenMix 30 (CCH)','Protaphan (CCH)','Detemir Gly (CCH)',
                               'NovoMix 70 NN2000','Actrapid','Aspart (CHH)',
                               'FLEX.LIRA 6MG 8.15 LI.BLUE CAP','FlexPen Aspart 100 3ML CCH',
                               'Liraglutid','Liraglutid lysBlKapsel (placebo)','Testmedium (CCH)',
                               'no product'], [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16], inplace=True)


#DF=DF[DF.ProductDescription==1]
X = DF.drop(['Error_Count','Line','DateTime_Out','ProductDescription','OutputGood','Feed'],axis=1)
attributeNames = X.columns.values.tolist()
#X = stats.zscore(X); #Normalize data
X = X.values
N, M = X.shape
#g=DF.ProductDescription

y = DF.ProductDescription.astype(str).astype(int)
#y = np.asarray(np.mat(y))
y = y.squeeze()
# Tree complexity parameter - constraint on maximum depth
tc = np.arange(1, 15, 1)

# K-fold crossvalidation
K = 5
CV = model_selection.KFold(n_splits=K,shuffle=True)

# Initialize variable
Error_train = np.empty((len(tc),K))
Error_test = np.empty((len(tc),K))
besterror_t = 1e100
Error_dectree = np.empty((K,1))
Errors_s = np.empty((K,1))
k=0
for train_index, test_index in CV.split(X,y):
    print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))

    # extract training and test set for current CV fold
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]

    for train_index, test_index in CV.split(X_train,y_train ):
        print('Computing CV fold: {0}/{1}..'.format(k+1,K))

        # extract training and test set for current CV fold
        X_train_t, y_train_t = X[train_index,:], y[train_index]
        X_test_t, y_test_t = X[test_index,:], y[test_index]

        for i, t in enumerate(tc):
            # Fit decision tree classifier, Gini split criterion, different pruning levels
            dtc_t = tree.DecisionTreeClassifier(criterion='gini', max_depth=t)
            dtc_t = dtc_t.fit(X_train_t,y_train_t.ravel())
            y_est_test_t = dtc_t.predict(X_test_t)
            y_est_train_t = dtc_t.predict(X_train_t)

            # Evaluate misclassification rate over train/test data (in this CV fold)
            misclass_rate_test = sum(np.abs(y_est_test_t - y_test_t)) / float(len(y_est_test_t))
            misclass_rate_train = sum(np.abs(y_est_train_t - y_train_t)) / float(len(y_est_train_t))
            Error_test[i,k], Error_train[i,k] = misclass_rate_test, misclass_rate_train

    # mean af alle 5 test modelkompleksivitet
    Error_test_mean = Error_test.mean(1)
    # finder til hvilket t, som har den laveste mean
    max_depth_t = np.unravel_index(Error_test_mean.argmin(), Error_test_mean.shape)[0]+tc[0]

    #kan evt finde den t, som har den laveste af alle - tage ikke hensyn til mean?
    #max_depth_t = np.unravel_index(Error_test.argmin(), Error_test.shape)[0]+1




    dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth=max_depth_t)
    model_dectree = dtc.fit(X_train, y_train)
    y_dectree = model_dectree.predict(X_test)

    #Errors_s[k] = np.power(y_dectree-y_test,2).sum().astype(float)/y_test.shape[0]
    #Error_dectree[k] = 100*(y_dectree!=y_test).sum().astype(float)/len(y_test)

    k+=1


f = figure()
boxplot(Error_test.T, positions=tc, showmeans=True)
xlabel('Model complexity (max tree depth)')
ylabel('Test error across CV folds, K={0})'.format(K))
title("Boxplot with mean")

f = figure()
plot(tc, Error_train.mean(1))
plot(tc, Error_test.mean(1))
xlabel('Model complexity (max tree depth)')
ylabel('Error (misclassification rate, CV K={0})'.format(K))
legend(['Error_train','Error_test'])

show()



dtc_fit = tree.DecisionTreeClassifier(criterion='gini', max_depth=max_depth_t)
dtc_fit = dtc.fit(X,y)

# Export tree graph for visualization purposes:
# (note: you can use i.e. Graphviz application to visualize the file)
out = tree.export_graphviz(dtc_fit, out_file='Novomix.gvz', feature_names=attributeNames)
DF.to_csv('DFpennr.csv',sep='\t')




