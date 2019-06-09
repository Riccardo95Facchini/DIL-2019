import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

#EVERY TIME THE DATASET IS RETRIEVED FROM GITHUB

input_file = 'https://raw.githubusercontent.com/lcphy/Digital-Innovation-Lab/master/bank-full.csv'
dataset = pd.read_csv(input_file, sep=';', header = 0)

dataset.head()

#DELETE NEXT CALLS DATA

dataset = dataset.drop("contact", axis=1)
dataset = dataset.drop("day", axis=1)
dataset = dataset.drop("month", axis=1)
dataset = dataset.drop("duration", axis=1)
dataset = dataset.drop("campaign", axis=1)
dataset = dataset.drop("pdays", axis=1)
dataset = dataset.drop("previous", axis=1)
dataset = dataset.drop("poutcome", axis=1)

dataset.head()

#FEATURE ENGINEERING

cleanup_nums = {"marital":     {"married": 1, "single": 0, "divorced":-1},
                "education": {"primary": 1, "secondary": 2, "tertiary": 3},
               "default":     {"yes": 1, "no": 0},
               "housing":     {"yes": 1, "no": 0},
               "loan":     {"yes": 1, "no": 0},
               "y":     {"yes": 1, "no": 0}}

dataset.replace(cleanup_nums, inplace=True)
dataset.head()

dataset.dtypes

dataset = dataset[dataset.job != 'unknown']
dataset = dataset[dataset.education != 'unknown']
dataset['education'] = dataset['education'].astype(int)

#COLLERATION MATRIX

plt.figure(figsize=(12,10))
cor = dataset.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

#CLASSIFIFICATION

X = dataset.iloc[:, 0:7]
y = dataset.iloc[:, 7]

X = pd.get_dummies(X, columns=["job"], prefix=["job"])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

#DECISION TREE

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

clf_dt = DecisionTreeClassifier()
clt_dt = clf_dt.fit(X_train,y_train)

esito = clf_dt.predict(X_test)

target_names = ['NOT-sub', 'Subscribed']
print(classification_report(y_test, esito,target_names=target_names))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, esito)
print(cm)

plt.hist(esito)

#RANDOM FOREST

from sklearn.ensemble import RandomForestClassifier

clf_dt = RandomForestClassifier()
clt_dt = clf_dt.fit(X_train,y_train)

esito = clf_dt.predict(X_test)

target_names = ['NOT-sub', 'Subscribed']
print(classification_report(y_test, esito,target_names=target_names))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, esito)
print(cm)

plt.hist(esito)

# K-NEAREST NEIGHBOURS

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# TRAINING - TEST
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# SCALING
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# FITTING
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# PREDICTION
y_pred = classifier.predict(X_test)

# CONFUSION MATRIX
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

target_names = ['NOT-sub', 'Subscribed']
print(classification_report(y_test, y_pred,target_names=target_names))

print(cm)

plt.hist(y_pred)

#UNDERSAMPLING

from sklearn.utils import resample

dataset_sample = pd.get_dummies(dataset, columns=["job"], prefix=["job"])

#SPLIT FEATURE AND TARGET
y = dataset_sample.y
X = dataset_sample.drop('y', axis=1)

#TRAIN TEST
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

X = pd.concat([X_train, y_train], axis=1)

#SELECTING TARGET CLASSES

not_sub = X[X.y==0]
sub = X[X.y==1]

not_sub_downsampled = resample(not_sub,
                                replace = False,
                                n_samples = len(sub),
                                random_state = 27)

# COMBINE MINORITY AND DOWNSAMPLED MAJORITY
downsampled = pd.concat([not_sub_downsampled, sub])

#DECISION TREE

y_train = downsampled.y
X_train = downsampled.drop('y', axis=1)

clf_dt = DecisionTreeClassifier()
clt_dt = clf_dt.fit(X_train,y_train)

esito = clf_dt.predict(X_test)

target_names = ['NOT-sub', 'Subscribed']
print(classification_report(y_test, esito,target_names=target_names))

#RANDOM FOREST
y_train = downsampled.y
X_train = downsampled.drop('y', axis=1)

clf_dt = RandomForestClassifier()
clt_dt = clf_dt.fit(X_train,y_train)

esito = clf_dt.predict(X_test)

target_names = ['NOT-sub', 'Subscribed']
print(classification_report(y_test, esito,target_names=target_names))

#SMOTE - DECISION TREE

from imblearn.over_sampling import SMOTE

#SPLIT FEATURE TARGET
y = dataset_sample.y
X = dataset_sample.drop('y', axis=1)

#TRAIN TEST
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#SMOTE
sm = SMOTE(random_state=27, ratio=1.0)
X_train, y_train = sm.fit_sample(X_train, y_train)

clf_dt = DecisionTreeClassifier()

#FIT
smote = clf_dt.fit(X_train,y_train)

#PREDICITON
smote_pred = smote.predict(X_test)

target_names = ['NOT-sub', 'Subscribed']
print(classification_report(y_test, smote_pred,target_names=target_names))

#SMOTE - RANDOM FOREST

from imblearn.over_sampling import SMOTE

y = dataset_sample.y
X = dataset_sample.drop('y', axis=1)

# setting up testing and training sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

sm = SMOTE(random_state=27, ratio=1.0)
X_train, y_train = sm.fit_sample(X_train, y_train)

clf_dt = RandomForestClassifier()

smote = clf_dt.fit(X_train,y_train)

smote_pred = smote.predict(X_test)

target_names = ['NOT-sub', 'Subscribed']
print(classification_report(y_test, smote_pred,target_names=target_names))

#RECAP on RECALL

x = np.arange(3)
plt.bar(x-0.2, [31,65,37], width=0.2, color='b', align='center', label='DT')
plt.bar(x, [18,61,32], width=0.2, color='r', align='center', label='RF')
plt.xticks(x-0.1, ['Normal','Under','Smote'])
plt.legend(loc='upper right')

#RECAP on F1

x = np.arange(3)
plt.bar(x-0.2, [31,26,32], width=0.2, color='b', align='center', label='DT')
plt.bar(x, [24,28,31], width=0.2, color='r', align='center', label='RF')
plt.xticks(x-0.1, ['Normal','Under','Smote'])
plt.legend(loc='lower right')