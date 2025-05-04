import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report
import joblib
import pickle

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')



# Only show up to 5 columns when printing a DataFrame
pd.set_option('display.max_columns', 12)

# To get some idea on the data.
print(train[['Survived','Name']].head())
print(train.info())
print(train.describe())
print(train.isnull().sum())

sns.countplot(x='Survived', hue = 'Sex',data=train)
plt.show()

sns.countplot(x='Survived', hue = 'Pclass',data=train)
plt.show()

# Data parsing

train['TestTrain'] = 'Train'
test['TestTrain'] = 'Test'


combined= pd.concat([train,test],sort=False) #No need to sort

# Handle missing values

combined['Age']=combined['Age'].fillna(combined['Age'].median())
combined['Fare']=combined['Fare'].fillna(combined['Fare'].median())
combined['Embarked']=combined['Embarked'].fillna(combined['Embarked'].mode()[0])

combined['Title'] = combined['Name'].str.extract(' ([A-Za-z]+)\\.', expand=False)
title_map={
    'Mlle':'Miss',
    'Ms':'Miss',
    'Mlle':'Miss',
    'Mme':'Mrs',
    'Don':'Sir',
    'Major':'Sir',
    'Col':'Sir',
    'Capt':'Sir',
    'Sir':'Sir',
    'Jonkheer':'Sir',
    'Rev':'Sir'
}
combined['Title'] = combined['Title'].replace(title_map)

combined['FamilySize'] = combined['SibSp'] + combined['Parch']+1

combined.drop(['Name','Ticket','Cabin'], axis=1, inplace=True)

combined['Sex']= combined['Sex'].map( {'female': 1, 'male': 0} )
combined = pd.get_dummies(combined, columns=['Embarked','Title'])

train_cleaned = combined[combined['TestTrain']=='Train'].drop('TestTrain',axis=1)
test_cleaned = combined[combined['TestTrain']=='Test'].drop('TestTrain',axis=1)

X = train_cleaned.drop(['Survived','PassengerId'],axis=1)
Y = train_cleaned['Survived']

X_train, X_val, y_train, y_val = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_val)
print(f"Mean squared error: {mean_squared_error(y_val, y_pred):.2f}")
print(f"Coefficient of determination: {r2_score(y_val, y_pred):.2f}")

lmodel= LogisticRegression()
lmodel.fit(X_train, y_train)

ly_pred = lmodel.predict(X_val)
print("Accuracy of logistic regression model: ", accuracy_score(y_val, ly_pred))
print(confusion_matrix(y_val, ly_pred))
print(classification_report(y_val, ly_pred))



# After training and before serializing:
feature_names = X_train.columns.tolist()
with open('model_features.pkl', 'wb') as f:
    pickle.dump(feature_names, f)

joblib.dump(lmodel, 'model.pkl')

