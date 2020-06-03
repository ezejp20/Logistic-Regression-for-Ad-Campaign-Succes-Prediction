# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 19:09:28 2019

@author: I519797
"""
# Load the needed libraries with aliases
import pandas as pd 
import math
import numpy as np
import datetime as dt
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True) 
data = pd.read_csv("sample.csv") 
#I want to group together the birth dates into rough segments 18-25,25-35,35-45,45-55,55+
#First I just take the years
data['BirthDate']=data['BirthDate'].str[-4:]
for i in range(0,len(data['BirthDate'])):
    if not isinstance(data['BirthDate'][i], (str)) and math.isnan(data['BirthDate'][i]):
        data['BirthDate'][i]="0"
        #Change all the invalid date forms into 0
    if "-" in data['BirthDate'][i]:
        data['BirthDate'][i]="0"
#I will change all the 1900 values to 0
data['BirthDate']=data['BirthDate'].str.replace('1900','0')
for i in range(0,len(data['BirthDate'])):
        #data['BirthDate']=data['BirthDate'].astype(int)
        if 1994<=int(data['BirthDate'][i])<=2002:
           data['BirthDate'][i]='18-25'
        elif 1984<=int(data['BirthDate'][i])<1994:
            data['BirthDate'][i]='26-35'
        elif 1974<=int(data['BirthDate'][i])<1984:
            data['BirthDate'][i]='36-45'
        elif 1964<=int(data['BirthDate'][i])<1974:
            data['BirthDate'][i]='46-55'
        elif 1954<=int(data['BirthDate'][i])<1964:
            data['BirthDate'][i]='56-65'
        elif 0<int(data['BirthDate'][i])<1954:
            data['BirthDate'][i]='65+'
data.rename(columns={'BirthDate':'Age Group'},inplace=True)
#%%
data['campaign_1'].value_counts()
sns.countplot(x='campaign_1', data=data, palette='hls')
plt.suptitle('Campaign 1 responses', fontsize=16)
plt.show()
count_no_sub = len(data[data['campaign_1']==0])
count_sub = len(data[data['campaign_1']==1])
pct_of_no_sub = count_no_sub/(count_no_sub+count_sub)
print("percentage of no subscription for campaign 1 is", pct_of_no_sub*100)
pct_of_sub = count_sub/(count_no_sub+count_sub)
print("percentage of subscription for campaign 1 is", pct_of_sub*100)
plt.savefig('Count Plot for Campaign 1')
data['campaign_2'].value_counts()
sns.countplot(x='campaign_2', data=data, palette='hls')
plt.suptitle('Campaign 2 responses', fontsize=16)
plt.show()
plt.savefig('Count Plot for Campaign 2')
count_no_sub = len(data[data['campaign_2']==0])
count_sub = len(data[data['campaign_2']==1])
pct_of_no_sub = count_no_sub/(count_no_sub+count_sub)
print("percentage of no subscription for campaign 2 is", pct_of_no_sub*100)
pct_of_sub = count_sub/(count_no_sub+count_sub)
print("percentage of subscription for campaign 2 is", pct_of_sub*100)
data['campaign_3'].value_counts()
sns.countplot(x='campaign_3', data=data, palette='hls')
plt.suptitle('Campaign 3 responses', fontsize=16)
plt.show()
count_no_sub = len(data[data['campaign_3']==0])
count_sub = len(data[data['campaign_3']==1])
pct_of_no_sub = count_no_sub/(count_no_sub+count_sub)
print("percentage of no subscription for campaign 3 is", pct_of_no_sub*100)
pct_of_sub = count_sub/(count_no_sub+count_sub)
print("percentage of subscription for campaign 3 is", pct_of_sub*100)
plt.savefig('Count Plot for Campaign 3')
data['campaign_5'].value_counts()
sns.countplot(x='campaign_5', data=data, palette='hls')
plt.suptitle('Campaign 5 responses', fontsize=16)
plt.show()
plt.savefig('Count Plot for Campaign 5')
count_no_sub = len(data[data['campaign_5']==0])
count_sub = len(data[data['campaign_5']==1])
pct_of_no_sub = count_no_sub/(count_no_sub+count_sub)
print("percentage of no subscription for campaign 5 is", pct_of_no_sub*100)
pct_of_sub = count_sub/(count_no_sub+count_sub)
print("percentage of subscription for campaign 5 is", pct_of_sub*100)
data['campaign_8'].value_counts()
sns.countplot(x='campaign_8', data=data, palette='hls')
plt.suptitle('Campaign 8 responses', fontsize=16)
plt.show()
plt.savefig('Count Plot for Campaign 8')
count_no_sub = len(data[data['campaign_8']==0])
count_sub = len(data[data['campaign_8']==1])
pct_of_no_sub = count_no_sub/(count_no_sub+count_sub)
print("percentage of no subscription for campaign 8 is", pct_of_no_sub*100)
pct_of_sub = count_sub/(count_no_sub+count_sub)
print("percentage of subscription for campaign 8 is", pct_of_sub*100)
#%%
#I now want to visualise the number of subscriptions grouped by job title
table=pd.crosstab(data['MaritalStatus'],data['campaign_1'])
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Marital Status vs Purchase Campaign 1')
plt.xlabel('Marital Status')
plt.ylabel('Proportion of Customers')
plt.savefig('marital_vs_pur_stack1')
table=pd.crosstab(data['MaritalStatus'],data['campaign_2'])
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Marital Status vs Purchase Campaign 2')
plt.xlabel('Marital Status')
plt.ylabel('Proportion of Customers')
plt.savefig('marital_vs_pur_stack2')
table=pd.crosstab(data['MaritalStatus'],data['campaign_3'])
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Marital Status vs Purchase Campaign 3')
plt.xlabel('Marital Status')
plt.ylabel('Proportion of Customers')
plt.savefig('marital_vs_pur_stack3')
table=pd.crosstab(data['MaritalStatus'],data['campaign_5'])
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Marital Status vs Purchase Campaign 5')
plt.xlabel('Marital Status')
plt.ylabel('Proportion of Customers')
plt.savefig('marital_vs_pur_stack5')
table=pd.crosstab(data['MaritalStatus'],data['campaign_8'])
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Marital Status vs Purchase Campaign 8')
plt.xlabel('Marital Status')
plt.ylabel('Proportion of Customers')
plt.savefig('marital_vs_pur_stack8')
#%%
#Next I created a new data frame containing only the data I wanted to use
#definitnely need to rename data frame every time otherwise im gonna have to run dat long ass code again and fuck that
#poo=data #this is a spare dataframe incase I mess up
#wee=data.drop(data.columns[[0, 1, 2,6,8,9,11,12,13,14,15,16,17,18]]) 
#dataGO=poo
#%%
#Creating dummy variables
dummy_rank_ctg = pd.get_dummies(data['CustomerTypeGroup'])
dummy_rank_cts=pd.get_dummies(data['CustomerSegment'])
dummy_rank_slsch=pd.get_dummies(data['SalesChannel'])
dummy_rank_gndr=pd.get_dummies(data['Gender'])
dummy_rank_age=pd.get_dummies(data['Age Group'])
dummy_rank_mar=pd.get_dummies(data['MaritalStatus'])
dummy_rank_nat=pd.get_dummies(data['Nationality'])
dummy_rank_occup=pd.get_dummies(data['Occupation'])
data3=pd.concat([data,dummy_rank_ctg,dummy_rank_cts,dummy_rank_slsch,dummy_rank_gndr, dummy_rank_age, dummy_rank_mar, dummy_rank_nat, dummy_rank_occup], axis=1)

#print (data4.head())
#%%
print (data3.head())
data4=data3.drop(['Unnamed: 0','SubscriberID','SnapshotDate'],axis=1)
data4=data4.drop(['campaign_4','campaign_6','campaign_7','campaign_9','campaign_10','campaign_11','campaign_12','campaign_13','campaign_14','campaign_15','campaign_16'], axis=1)
print (data4.head())
#%%
data5=data4.drop(['campaign_2','campaign_1','campaign_5','campaign_8'], axis=1)
#%%
data6=data5.drop(['CustomerTypeGroup','CustomerSegment','SalesChannel','Gender','Age Group', 'MaritalStatus','Nationality','Occupation'], axis=1)
#%%
data6=int(data6)
                   
#%%
#need to work out why this didn't work 
import pandas as pd
import statsmodels.api as sm
#import pylab as pl
#import numpy as np
data7['intercept'] = 1.0
train_cols = data7.columns[1:]
logit = sm.Logit(data7['campaign_3'], data.iloc[train_cols])
result = logit.fit()
print (result.summary())
#%%
#Change Uint8 to int
data7=data6.astype('int')
#%%
lr = LogisticRegression()
lr.fit(data.iloc[train_cols])
#%%
print(lr.coef_)
print(lr.intercept_)
#%%
y_pred = lr.predict(x_test)
#%%
confusion_matrix(y_test, y_pred)
#%%
print (data7.dtypes)
#%%
columnnames=data7.columns[1:]
X = data7[columnnames] # Features
y = data7['campaign_3'] # Target variable
#%%
from sklearn import *
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
#%%
# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
logreg.fit(X_train,y_train)

#
y_pred=logreg.predict(X_test)


#%%
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix
# import required modules
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#matplotlib inline
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))