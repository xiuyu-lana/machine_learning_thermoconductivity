
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#alloy=pd.read_excel('alloy_full_nomech_Sifix.xlsx')
alloy=pd.read_csv('alloy_processed.csv')

alloy = alloy[alloy.delta_T_ek < 50]

#alloy = alloy[alloy.term1 < 1] # Avoid problematic rows for the time being
alloy = alloy.reset_index() # Will mess up from deleted items if don't reset this
comp = alloy.loc[:, 'Al':'Zr'].div(alloy['Sum'],axis=0)
comp = comp.loc[:, (comp != 0).any(axis=0)]


rebuild = pd.DataFrame()
example = comp.loc[0,:] # Column vector of compositions of first alloy 
names = comp.columns # This shows that it works perfectly
namesij = np.array(list(combinations(names, 2)))
# use this to assign column names eventually

#  get X matrix and Y vector
X = pd.DataFrame()
for i in range(alloy.shape[0]):
    example = comp.loc[i,:] 
    subsets1 = np.array(list(combinations(example, 1)))
    subsets2 = np.array(list(combinations(example, 2)))
    
    Xij = subsets2[:,0] * subsets2[:,1]
    subsets = np.append(subsets1, Xij)
    
    X = pd.concat([X,pd.DataFrame(subsets).transpose()],ignore_index=True)

# Get Y vector
Y = 1/alloy['k_total(W/m-K)'] - alloy['term1']
# there are some really strange results! The "term1" column has some in the tens of thousands but most are like 0.003 and stuff
# Need to check this. Did I have the same problem for the non-interpolated version? 


#Y = Y[Y < 1] # Avoid problematic rows for the time being
#Y = Y[Y > -1] # Avoid problematic rows for the time being
# drop 162 and 156
#X = X.drop(162)
#X = X.drop(156)
#X = X.reset_index()
#Y = Y.reset_index()

# split into training and test sets 
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.utils import shuffle
#from sklearn.model_selection import train_test_split
ncvfolds = 5
rs = 0 # random state
#X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=1/ncvfolds, shuffle=False)



for j in range(2,11):
    ncvfolds = j
    X, Y, alloy = shuffle(X, Y, alloy, random_state=rs)
    Y=pd.DataFrame(Y)
    
    kf = KFold(n_splits=ncvfolds, shuffle=True,random_state=rs)
    percerravg = [0]*ncvfolds
    r2arr = [0]*ncvfolds
    i = 0
    
    for train_index, test_index in kf.split(X):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        k_train, k_test = alloy['k_total(W/m-K)'].iloc[train_index], alloy['k_total(W/m-K)'].iloc[test_index]
        term1_train, term1_test = alloy['term1'].iloc[train_index], pd.DataFrame(alloy['term1'].iloc[test_index])
        
        # Try doing everything is train/test
        #X_train, X_test = X, X
        #Y_train, Y_test = Y, Y
    
        #X = X_train # calculate omega hat on training data
        square = X_train.transpose().dot(X_train) # results is a square matrix 
        inverse = pd.DataFrame(np.linalg.pinv(square.values), square.columns, square.index)
        identitycheck = inverse.dot(square)
        # This is not really that close to the identity matrix. Isn't that bad? 
        # I deleted columns that were all zeros and that was better but still
        result = inverse.dot(X_train.transpose())
        Omega_hat = result.dot(Y_train)
        
        Y_hat = X_test.dot(Omega_hat)
        blah = pd.concat([Y_hat, term1_test],axis=1)
        blah['sum'] = blah.iloc[:,0] + blah['term1']
        k_hat = 1/(blah['sum'])
        #k_hat = pd.DataFrame
        err = k_test-k_hat
        
        #err = Y_test-Y_hat
        errabs = abs(err)
        erravg = np.average(errabs)
        errperc = erravg/k_test*100
        errpercavg = np.average(abs(errperc))
        percerravg[i] = errpercavg
        #print(errpercavg)
        errRMSE = np.linalg.norm(err)
        errMSE = np.linalg.norm(err)**2
        
        Yavg = np.average(Y_test)
        SS = np.sum((Y_test-Yavg)**2)
        RSS = np.sum(err**2)
        R2 = 1-RSS/SS
        #print(R2)
        
        R2sk = r2_score(Y_test, Y_hat)
        #print(R2sk)
        r2arr[i] = R2sk
        i = i+1
    R2 = np.average(r2arr)
    percerr = np.average(percerravg)
    
    fig = plt.figure(1, figsize=(8,5))
    label = str(ncvfolds) +  ' Folds' 
    xarray = [j]*j
    plt.scatter(xarray,r2arr, label=label) 
    #plt.scatter(j,R2, c='b',label=label) 
    plt.xlabel('# of Folds in Cross-Validation')
    plt.ylabel('R^2 Value')


    fig = plt.figure(2, figsize=(8,5))
    label = str(ncvfolds) +  ' Folds' 
 
    plt.scatter(xarray,percerravg, label=label) 
    #plt.scatter(j,percerr, c='b', label=label) 
    plt.xlabel('# of Folds in Cross-Validation')
    plt.ylabel('Percent Error Value')

plt.show()

