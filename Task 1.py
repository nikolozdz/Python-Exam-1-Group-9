import pandas as pd
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('creditcard.csv') # imports data

counts = dataset['Class'].value_counts()
print (counts)
X = dataset.drop('Class', axis=1)
y = dataset['Class']

# Splitting the data for training 67% and testing 33%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state= 0)



classifier = SVC(kernel="linear") # Fitting Linear SVM to the Training set
classifier.fit(X_train, y_train)


y_pred = classifier.predict(X_test) # Predicting the Test set results

# Calculating SVC Accuracy
print("SVM Accuracy is: ",metrics.accuracy_score(y_test,y_pred)*100)
print(classification_report(y_test,y_pred))



'''
RAWA ALBATTAWI Code

import numpy as np
import pandas as pd
import sklearn
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from pylab import rcParams

rcParams['figure.figsize'] = 14, 8
RANDOM_SEED = 42
LABELS = ["Normal", "Fraud"]


dataset = pd.read_csv('creditcard.csv')
print(dataset.head())
print(dataset.info())

#Create independent and Dependent Features
columns = dataset.columns.tolist()
# Filter the columns to remove dataset we do not want 
columns = [c for c in columns if c not in ["Class"]]
# Store the variable we are predicting 
target = "Class"
# Define a random state 
state = np.random.RandomState(42)
X = dataset[columns]
Y = dataset[target]
X_outliers = state.uniform(low=0, high=1, size=(X.shape[0], X.shape[1]))
# Print the shapes of X & Y
print(X.shape)
print(Y.shape)

# Check Null values in dataset
dataset.isnull().values.any()


#Visuilize unbalanced dataset as Normal transuction and fraud
count_classes = pd.value_counts(dataset['Class'], sort = True)
count_classes.plot(kind = 'bar', rot=0)
plt.title("Transaction Class Distribution")
plt.xticks(range(2), LABELS)
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.show()



## Get the Fraud and the normal dataset 
fraud = dataset[dataset['Class']==1]
normal = dataset[dataset['Class']==0]
print('Original Dataset:')
print('Fruad data:',fraud.shape,' - Normal data:',normal.shape)


# Implementing Undersampling for Handling Imbalanced Package and importing NearMiss module
#install model with command conda install -c conda-forge imbalanced-learn
# what will do is to down sampling the normal dataset to the amount of fruad dataset
# to get a balanced dataset like 50/50 %
from imblearn.under_sampling import NearMiss
# from imblearn import under_sampling
# from under_sampling import NearMiss
from imblearn import *
nm = imblearn.under_sampling.NearMiss(random_state=42)
X_res,y_res=nm.fit_sample(X,Y)
print('New Daataset after down sampling:')
print('Fruad data:',X_res.shape,' - Normal data:',y_res.shape)


# Print the count for normal and fraud data in dataset
from collections import Counter
print('Original dataset shape {}'.format(Counter(Y)))
print('Resampled dataset shape {}'.format(Counter(y_res)))


'''

