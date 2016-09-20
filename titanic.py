#!/usr/bin/python

import sys
import pandas as  pd
import csv as csv
import numpy
from feature_selection import new_features
from algorithms import gaussNB,LogReg,RandForest,LinearS, DTree, GridSearch
from tester import test_classifier

### 1. Feature Lists

features_list = ['Survived','nofamily3rd','fem_1st_fareover29_under65','fare_age_combo','Fare','Pclass','Sex','female1st2nd']
test_features_list = ['nofamily3rd','fem_1st_fareover29_under65','fare_age_combo','Fare','Pclass','Sex','female1st2nd']


## the data has been cleaned already (http://www.bmdata.co.uk/titanic_code.html) and converted to numeric
## decision has been made to remove 3 features: Name, Ticket and Cabin

## 2. Load cleaned data
titan_test = pd.read_csv('clean_titanictest.csv',header=0)
titan_train = pd.read_csv('clean_titanictrain.csv',header=0)

## 3. Create New Features
titan_test = new_features(titan_test)
titan_train = new_features(titan_train)


## 4. prepare dataframes
## put  features  in the right order for training data
titan_train = titan_train[features_list]
## store passenger ids (for training labels) and delete from test data
ids = titan_test['PassengerId'].values
titan_test.pop('PassengerId')
titan_test = titan_test[test_features_list]

#5. Convert to numpy arrays
train_data = titan_train.values
test_data = titan_test.values

#set the features and labels for the training data
features = train_data[0::,1::]
labels = train_data[0::,0]


#6. Grid Search CV - need to do train/test split first
#from sklearn import cross_validation
#features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.3, random_state=42)
#GridSearch(scaler,kbest,features_train,labels_train,features_test,labels_test)


#7. SelectKBest
from sklearn.feature_selection import SelectKBest

x = 0
klevel = 'all'
kbest = SelectKBest(k=klevel)
kbest.fit_transform(features,labels)

for f in features_list:
    if x > 0:
        print(f,'KBEST',kbest.scores_[x-1])
    x = x + 1

#7. Scaler (doesn't make a difference for Decision Trees but using anyway)
from sklearn.preprocessing import MinMaxScaler


scaler =MinMaxScaler()
#comment out to test out different algorithms (see algorithms.py)
#clf = gaussNB(scaler,kbest)
#clf = DTree(scaler,kbest)
#clf = LinearS(scaler,kbest)
#clf = LogReg(scaler,kbest)
clf = RandForest(scaler,kbest)
#StratifiedShuffleSplit tester (from Enron exercise)
test_classifier(features,labels, clf)


#code for testing algorithm with train/test split instead

#clf.fit(features_train,labels_train)

#pred = clf.predict(features_test)
#from sklearn.metrics import accuracy_score
#acc = accuracy_score(pred, labels_test)
#from sklearn.metrics import precision_score
#precision = precision_score(pred,labels_test)
#from sklearn.metrics import recall_score
#rec = recall_score(pred, labels_test)
#print ('accuracy:',acc)
#print('precision:',precision)
#print('recall:',rec)


#code to dump results to csv file for upload onto kaggle
clf.fit(features, labels)
predictions = clf.predict(test_data).astype(int)

predictions_file = open("myfirstforest.csv", "w")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, predictions))
predictions_file.close()
print ('Done.')
