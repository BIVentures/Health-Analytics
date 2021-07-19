# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 12:28:40 2021
File: Detecting Parkinson’s Disease with Machine Learning Algorithm XGBoost
@Author:BIVentures
Disclaimer: Data used in this file is publicly available

Context: Parkinson's Disease is a degenerative disorder which targets 
the central nervous system. It affects the dopamine-producing 
neurons found in the brain which hampers movement, primarily, 
of  limbs  in  the  body.  There  is  no  standard  test  to  diagnose 
Parkinson’s Disease, a condition that affects up to one million 
people in the US [1]. Symptoms develop slowly over the years 
which include tremors in hands, unbalancing while walking and 
even an altered taste in smell in a few cases as per Parkinson.org. 
As the disease advances the symptoms typically become more 
severe  and  weakening.  The  disease  also  causes  non-motor 
symptoms  which  often  appear  before  a  person  experiences 
motor symptom and can prove to be more troublesome for some.  
 
Non-motor  symptoms  include  fatigue,  excessive  saliva, 
constipation,  vision  and  dental  problems  and  lack  of  facial 
expressions. Another interesting observation in PD patients is 
their  inability  to  generate  high  force  levels  in  limbs  during 
locomotion.
"""

#### Import Libraries ####
import sys
import os
import numpy as np
import pandas as pd
from os import path
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix



#### Identify & Change Working Directory ####
print('System Path: ', sys.executable)
print('\n\n')
os.chdir('C:/Users/windo/Desktop/PythonVentures/Parkinsons')
currentDirectory=os.getcwd()
print(currentDirectory)	
print('\n')	

#### Read the data into a DataFrame ####
df=pd.read_csv('parkinsons.csv')
print(df.head())

#### Get the features and labels ####
# The features are all the columns, except 'status'
# The labels are those in the 'status' column.
features=df.loc[:,df.columns!='status'].values[:,1:]
labels=df.loc[:,'status'].values

#### Get the count of both labels for each row, where (0 and 1) in labels #### 
# The ‘status’ column has values 0 and 1 as labels 
print("Count of Label 1:",labels[labels==1].shape[0])
print("Count of Label 0:",labels[labels==0].shape[0])
print('\n')	

#### Scale the features to between -1 and 1 #### 
# Initialize a MinMaxScaler and scale the features to between -1 and 1 to normalize them. 
# The MinMaxScaler transforms features by scaling them to a given range. 
# The fit_transform() method fits to the data & then transforms it.
# This  significantly  reduced  data  by scaling  between  (-1,1)  
# This increases  data  consistency & reduces  computational  power  required  
# Also  keeping singularity of the data intact. 
scaler=MinMaxScaler((-1,1))
x=scaler.fit_transform(features)
y=labels

#### Build Classifier Model by Splitting the dataset #### 
# Initialize an XGBClassifier and train the model. 
# This classifies using eXtreme Gradient Boosting- 
# Using gradient boosting algorithms for modern data science problems. 
# XGBClassifier falls under the category of Ensemble Learning in ML. 
# Use XGB Classifier to train and predict using many models to produce one superior output
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2, random_state=7)

#### Train the model #### 
model=XGBClassifier(use_label_encoder = False,eval_metric='mlogloss')
model.fit(x_train,y_train)

#### Calculate the % accuracy #### 
y_pred=model.predict(x_test)
y_pred_value=round((accuracy_score(y_test, y_pred)*100),2)
print("Accuracy Rate: ", y_pred_value)
print("Percentage Accuracy: ",format(y_pred_value, ',.2f') + '%')
print('\n')

# IMPORTANT: first argument is true values, second argument is predicted values
# this produces a 2x2 numpy array (matrix)
labels = ['True', 'False']
print("Confusion Matrix 2x2 as below: \n")
cm=metrics.confusion_matrix(y_test, y_pred)
print(cm)
print("Confusion matrix:\n%s" % cm)




fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt     

ax= plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', ax=ax);  


# labels, title and ticks
ax.set_xlabel('Predicted labels');
ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['True', 'False']); 
ax.yaxis.set_ticklabels(['False', 'True']);