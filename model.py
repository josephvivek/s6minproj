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

# Select only 'TRANSFER' and 'CASH_OUT' transactions
X = df.loc[(df.type == 'TRANSFER') | (df.type == 'CASH_OUT')]

# Define features and target variable
Y = X['isFraud']
del X['isFraud']

# Drop irrelevant columns
X = X.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1)

# Binary encoding of 'type' column
X.loc[X.type == 'TRANSFER', 'type'] = 0
X.loc[X.type == 'CASH_OUT', 'type'] = 1
X['type'] = X['type'].astype(int)

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


# Evaluate accuracy on test set
predictions = clf.predict(testX)
accuracy = accuracy_score(testY, predictions)
print('Accuracy Score =', accuracy)

# Define input values for prediction
input_values = [1, 'TRANSFER', 800.0, 1500.0, 1500, 0, 800.0, 0, 0]

# Preprocess input values
input_df = pd.DataFrame([input_values], columns=X.columns)
input_df.loc[input_df.type == 'TRANSFER', 'type'] = 0
input_df.loc[input_df.type == 'CASH_OUT', 'type'] = 1
input_df['type'] = input_df['type'].astype(int)
input_df['errorBalanceOrig'] = input_df.newBalanceOrig + input_df.amount - input_df.oldBalanceOrig
input_df['errorBalanceDest'] = input_df.oldBalanceDest + input_df.amount - input_df.newBalanceDest

# Predict class
predicted_class = clf.predict(input_df)
print('Predicted Class:', predicted_class)


