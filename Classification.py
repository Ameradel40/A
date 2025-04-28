"""""""""""
Classification

"""""""""""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

iris = load_iris()


iris.feature_names // To call feature names of the dataset

Data_iris = iris.data // To copy data in Data_iris

Data_iris = pd.DataFrame(Data_iris, columns = iris.feature_names)// Call feature names and put in the columns, this to make dataframe


Data_iris['label'] = iris.target // To add target or classes to our dataset


plt.scatter(Data_iris.iloc[:,2], Data_iris.iloc[:,3], c = iris.target )// To visualize data, select all raws and and column 2 this is x and y should be our label
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.show()

x = Data_iris.iloc[:,0:4] // To select X, all columns
y = Data_iris.iloc[:,4] // To select Y, only target column


"""""""""""""""
k-NN Classifier

"""""""""""""""

from sklearn.neighbors import KNeighborsClassifier

kNN = KNeighborsClassifier(n_neighbors = 6, metric = 'minkowski', p = 1)

kNN.fit(x,y)

x_N = np.array([[5.6,3.4,1.4,0.1]])

kNN.predict(x_N)

x_N2 = np.array([[7.5,4,5.5,2]])

kNN.predict(x_N2)


from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, train_size = 0.8,
                                                    random_state = 88, shuffle= True,
                                                    stratify=y)


from sklearn.neighbors import KNeighborsClassifier

kNN = KNeighborsClassifier(n_neighbors = 50, metric = 'minkowski', p = 1)

kNN.fit(X_train,y_train)

predicted_types = kNN.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy_score(y_test,predicted_types)


"""""""""""""""
Decision Tree Classifier

"""""""""""""""

from sklearn.tree import DecisionTreeClassifier // To import DT classifier
from sklearn.metrics import accuracy_score      // To calculate accuracy matric

Dt = DecisionTreeClassifier()

from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, train_size = 0.8,
                                                    random_state = 88, shuffle= True,
                                                    stratify=y)

Dt.fit(X_train,y_train) 			// To fit our training dataset

Predicted_types_Dt = Dt.predict(X_test)		// To predict Test data

accuracy_score(y_test, Predicted_types_Dt)	// y_test is acual testing data and Predicted_types_Dt is predicted to calculate accuracy


from sklearn.model_selection import cross_val_score
Scores_Dt = cross_val_score(Dt, x, y, cv = 10)	// Dt is model

"""""""""""""""







