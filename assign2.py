# -*- coding: utf-8 -*-
"""
Created on Sun Mar 05 11:07:10 2017

@author: Induja
"""

import numpy
import pandas
from collections import defaultdict
from sklearn import linear_model, datasets

data = pandas.read_csv("C:\Users\Induja\Documents\CSE 258\Assignment 2\h1b_kaggle.csv")
data.rename(columns={'Unnamed: 0' : 'INDEX'},inplace=True)
data = data[pandas.notnull(data['CASE_STATUS'])]
data = data.drop(data[data.CASE_STATUS == 'REJECTED'].index)
data = data.drop(data[data.CASE_STATUS == 'INVALIDATED'].index)
data = data.drop(data[data.CASE_STATUS == 'PENDING QUALITY AND COMPLIANCE REVIEW - UNASSIGNED'].index)
#location_df = data['WORKSITE'].apply(lambda x: pandas.Series(x.split(',')))
threshold = 0.0001
#for col in df:
col = 'EMPLOYER_NAME'
counts = data[col].value_counts(normalize=True)
data = data.loc[data[col].isin(counts[counts > threshold].index), :]
data['CASE_STATUS'] = data['CASE_STATUS'].map({'CERTIFIED': 1, 'DENIED': 2, 'WITHDRAWN': 3, 'CERTIFIED-WITHDRAWN':4})
data['FULL_TIME_POSITION'] = data['FULL_TIME_POSITION'].map({'Y': 1, 'N': 0})

se = pandas.Series(range(len(data)))
data['INDEX'] = se.values
data = data.set_index('INDEX')

states = []
for row in data.iterrows():
    st = row[1]['WORKSITE'].split(',')[-1][1:]
    states.append(st)
s = pandas.Series(states)
data['STATES'] = s.values

#%%  
state_map = {
             'ALABAMA': 1, 
             'ALASKA': 2, 
             'ARIZONA': 3, 
             'ARKANSAS': 4, 
             'CALIFORNIA': 5,
             'COLORADO': 6, 
             'CONNECTICUT': 7, 
             'DELAWARE': 8, 
             'DISTRICT OF COLUMBIA': 9,
             'FLORIDA': 10, 
             'GEORGIA': 11, 
             'HAWAII': 12, 
             'IDAHO': 13, 
             'ILLINOIS': 14, 
             'INDIANA': 15,
             'IOWA': 16, 
             'KANSAS': 17, 
             'KENTUCKY': 18, 
             'LOUISIANA': 19, 
             'MAINE': 20, 
             'MARYLAND': 21,
             'MASSACHUSETTS': 22, 
             'MICHIGAN': 23, 
             'MINNESOTA': 24, 
             'MISSISSIPPI': 25, 
             'MISSOURI': 26,
             'MONTANA': 27, 
             'NEBRASKA': 28, 
             'NEVADA': 29, 
             'NEW HAMPSHIRE': 30,
             'NEW JERSEY': 31, 
             'NEW MEXICO': 32, 
             'NEW YORK': 33, 
             'NORTH CAROLINA': 34,
             'NORTH DAKOTA': 35, 
             'OHIO': 36, 
             'OKLAHOMA': 37, 
             'OREGON': 38, 
             'PENNSYLVANIA': 39,
             'PUERTO RICO': 40, 
             'RHODE ISLAND': 41, 
             'SOUTH CAROLINA': 42, 
             'SOUTH DAKOTA': 43,
             'TENNESSEE': 44, 
             'TEXAS': 45, 
             'UTAH': 46, 
             'VERMONT': 47, 
             'VIRGINIA': 48, 
             'WASHINGTON': 49,
             'WEST VIRGINIA': 50, 
             'WISCONSIN': 51, 
             'WYOMING': 52}
             
def feature(datum):
    feat = [0]*54
    feat[0] = 1
    feat[1] = int(datum['YEAR'])
    feat[state_map[datum['STATES']]+1] = 1
#    feat = [1, int(datum['YEAR']), state_map[datum['STATES']]]
    return feat

X_train = [feature(d[1]) for d in data.loc[:500000,:].iterrows() if d[1]['STATES'] != 'NA']
y_train = [d[1]['CASE_STATUS'] for d in data.loc[:500000,:].iterrows() if d[1]['STATES'] != 'NA']
X_test =  [feature(d[1]) for d in data.loc[500000:1000000,:].iterrows() if d[1]['STATES'] != 'NA'] 
y_test =  [d[1]['CASE_STATUS'] for d in data.loc[500000:1000000,:].iterrows() if d[1]['STATES'] != 'NA']            
#X_train = data.loc[:500000, 'YEAR'].tolist()
#y_train = data.loc[:500000,'CASE_STATUS'].tolist()
#X_test = data.loc[500000:1000000, 'YEAR'].tolist()
#y_test = data.loc[500000:1000000,'CASE_STATUS'].tolist()
#for index, item in enumerate(X_train):
#    X_train[index] = [1, item]
#for index, item in enumerate(X_test):
#    X_test[index] = [1, item]  
logreg = linear_model.LogisticRegression(C=10, multi_class='multinomial', solver='lbfgs')
logreg.fit(X_train, y_train)
pred = logreg.predict(X_test)
