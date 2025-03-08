	
"""
Pandas

"""

import pandas as pd

"""
Series

"""

Age = pd.Series([10,20,30,40],index=['age1','age2','age3','age4'])

Age.age3 // to select only age3

Filtered_Age = Age[Age>10]


# Calling Values of the Series
Age.values


# Calling Indices of the Series
Age.index

Age.index = ['A1','A2','A3','A4']

Age.index

"""""""""
DataFrame

"""""""""

import numpy as np

DF = np.array([[20,10,8],[25,8,10],[27,5,3],[30,9,7]])

#to create labels for dataset (for raws and culumns)
Data_Set = pd.DataFrame(DF)

#change labels from 1,2,3,4 to S1, S2, S3, S4
Data_Set = pd.DataFrame(DF,index = ['S1','S2','S3','S4'])

#set frame for raws and culumns 
Data_Set = pd.DataFrame(DF,index = ['S1','S2','S3','S4'],columns = ['Age','Grade1','Grade2'])

#Add a culumn
Data_Set['Grade3'] = [9,6,7,10]
==========================================


#extract a set of data or raw S2
Data_Set.loc['S2']

#not correct it is for labele not inded
Data_Set.loc[1][3]

#to select data from raw 1 and culumn 3
Data_Set.iloc[1][3]
Data_Set.iloc[1,3]


#select data from all raw culumn 0
Data_Set.iloc[:,0]

#select data from all raw culumn 3
Data_Set.iloc[:,3]

Filtered_Data = Data_Set.iloc[:,1:3]
=================================
#drop a culumn, axis=1 is used for culumn
Data_Set.drop('Grade1',axis=1)

# number 10 replaced by 12
Data_Set = Data_Set.replace(10,12)


replaced 12 to 10 and 9 to 30
Data_Set = Data_Set.replace({12:10, 9:30})

# check 3 firsat raws
Data_Set.head(3)


# check 2 last raws 
Data_Set.tail(2)

# to sort values of culumn Grade 1
Data_Set.sort_values('Grade1',ascending=True)

#axis=0 for raws and axis=1 for culumns // To sort index of raws
Data_Set.sort_index(axis=0, ascending = False)


# to read dataset
Data = pd.read_csv('Data_Set.csv')




