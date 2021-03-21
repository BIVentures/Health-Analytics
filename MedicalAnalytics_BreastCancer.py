# -*- coding: utf-8 -*-
"""
Machine Learning on Breast Cancer
Created on Sun Mar 21 21:55:00 2021
@author: BIVentures
"""

#importing the libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

pd.set_option('display.max_columns', 100)

os.chdir('C:/DirectoryPathHere/')

#importing our cancer dataset
df = pd.read_csv('cancer.csv',index_col= None, na_values='?')

#Get X and Y
X = df.iloc[:,1:31].values
Y = df.iloc[:,:31].values

#print top 5 rows
print(df.head())
print('\n')

#print  the dimensions of the data set using the panda dataset shape attribute
print("Cancer data set dimensions : {}".format(df.shape))

#Missing or Null Data points
print(df.isnull().sum())
print(df.isna().sum())


#### Drop the column with all missing values (na, NAN, NaN) ####
#NOTE: This drops the column Unnamed: 32 column
df = df.dropna(axis=1)
print(df.head())
#Get a count of the number of 'M' & 'B' cells
df['diagnosis'].value_counts()
#Visualize this count
# Import seaborn
import seaborn as sns 
sns.countplot(df['diagnosis'], label='diagnosis')

#### Standardizing multiple variables ####
# y includes diagnosis column with M or B values
y = df.diagnosis# drop the column 'id' as it is does not convey any useful info
# drop diagnosis since we are separating labels and features 
list = ['id','diagnosis']# X includes our features
X = df.drop(list,axis = 1)# get the first ten features
data_dia = y
data = X
data_std =(data-data.mean())/(data.std()) 
# standardization
# get the first 10 features
data = pd.concat([y,data_std.iloc[:,0:10]],axis=1)
data = pd.melt(data,id_vars='diagnosis',
 var_name='features',
 value_name='value')# make a violin plot
plt.figure(figsize=(10,10))
sns.violinplot(x='features', y='value', hue='diagnosis', data=data,split=True, inner='quart')
plt.xticks(rotation=90)

#sns.kdeplot(data=data_std)

#### See Patterns in Large Data ####
#correlation map
f,ax = plt.subplots(figsize=(18, 18))
matrix = np.triu(X.corr())
#sns.heatmap(X.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax, mask=matrix)


# create boxplots for texture mean vs diagnosis of tumor
plot = sns.boxplot(x='diagnosis', y='texture_mean', data=df, showfliers=False)
plot.set_title('Graph of texture mean vs diagnosis of tumor')

#### Pair Plot ####
#radius = df[['radius_mean','radius_se','radius_worst','diagnosis']]
#sns.pairplot(radius, hue='diagnosis',palette="husl", markers=["o", "s"],size=4)
#sns.pairplot(radius,kind="reg",size=4)

print('\n')
print(df['diagnosis'].value_counts())

"""
# make a new dataframe with only the desired feature for t test
from scipy import stats  
newdf = pd.DataFrame(data=df[['area_worst', 'diagnosis']])
new_d = newdf.set_index('diagnosis')
stats.ttest_ind(new_d.loc['M'], new_d.loc['B'])


# Move the reponse variable "diagnosis" to the end of the dataframe
end = df['diagnosis']
df.drop(labels=['diagnosis'], axis=1,inplace = True)
df.insert(30, 'diagnosis', end)
df.head()

def categorical_to_numeric_diagnosis(x):
    if x=='M':
        return 1
    if x=='B':
        return 0

df['diagnosis']= df['diagnosis'].apply(categorical_to_numeric_diagnosis)
df["diagnosis"].value_counts()

colors = np.array('b g r c m y k'.split()) #Different colors for plotting

fig,axes = plt.subplots(nrows =15,ncols=2, sharey=True,figsize = (15,50))
plt.tight_layout()
row = 0
iteration = 0
for j in range(0,len(df.columns[:-1])):
    iteration+=1
    if(j%2==0):
        k = 0
    else:
        k = 1
    sns.distplot(df[df.columns[j]],kde=False,hist_kws=dict(edgecolor="w", linewidth=2),color = np.random.choice(colors) ,ax=axes[row][k])
    if(iteration%2==0):
        row+=1
        plt.ylim(0,200)"""

##Print df in pretty table format
#from tabulate import tabulate
#print(tabulate(df, headers='keys', tablefmt='simple'))
#print(df.to_markdown()) 
# render dataframe as html
html = df.to_html()
print(html)

#write table to html
text_file = open("index.html", "w")
text_file.write(html)
text_file.close()