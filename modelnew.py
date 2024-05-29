import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost.sklearn import XGBClassifier
import pickle

# Read the dataset
df = pd.read_csv('paysim_dataset.csv')

# Rename columns for consistency
df = df.rename(columns={'oldbalanceOrg':'oldBalanceOrig', 'newbalanceOrig':'newBalanceOrig', \
                        'oldbalanceDest':'oldBalanceDest', 'newbalanceDest':'newBalanceDest'})

# Convert object columns to string
object_columns = df.select_dtypes(include=['object']).columns
df[object_columns] = df[object_columns].astype(str)

X = df

# Define features and target variable
Y = X['isFraud']
del X['isFraud']

# Drop irrelevant columns
X = X.drop(['nameOrig', 'nameDest'], axis=1)

# Calculate errorBalanceOrig and errorBalanceDest
X['errorBalanceOrig'] = X.newBalanceOrig + X.amount - X.oldBalanceOrig
X['errorBalanceDest'] = X.oldBalanceDest + X.amount - X.newBalanceDest
print(X.columns)

# Split data into train and test sets
randomState = 5
trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.2, random_state=randomState)

# Train XGBoost classifier
weights = (trainY == 0).sum() / (1.0 * (trainY == 1).sum())
clf = XGBClassifier(max_depth=3, scale_pos_weight=weights, n_jobs=4)
clf.fit(trainX, trainY)

pickle.dump(clf,open('model.pkl','wb'))
