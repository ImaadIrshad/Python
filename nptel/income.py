# work with dataframes with pandas
import pandas as pd

# perform numerical operations with numpy
import numpy as np

# visualize data with seaborn
import seaborn as sns

# partition the data with sklearn
from sklearn.model_selection import train_test_split

# for logistic regression using sklearn library
from sklearn.linear_model import LogisticRegression

# for performance metrics - accuracy score and confusion matrix
from sklearn.metrics import accuracy_score, confusion_matrix

# importing data
data_income = pd.read_csv('income.csv')

# copy of original data
data = data_income.copy()

# Exploratory data analysis
# getting to know the data, data preprocessing (missing values), cross tables and data visualization

# check variables datatype
print(data.info())

# check for missing values
data.isnull()

print('Data columns with null values:\n', data.isnull().sum())
# currently no missing values

# summary of numerical variables
summary_num = data.describe()
print(summary_num)

# summary of categorical variables
summary_cate = data.describe(include = "O")
# "O" refers to object
print(summary_cate)

# frequency of each categories
data['JobType'].value_counts()
data['occupation'].value_counts()

# checking for unique classes (what are "?")
print(np.unique(data['JobType']))
print(np.unique(data['occupation']))

# there exists ' ?' instead of nan

# we go back and read the data by including "na_values[' ?']" to consider ' ?' as nan
data = pd.read_csv('income.csv', na_values=[" ?"])

# data pre-processing
data.isnull().sum()

missing = data[data.isnull().any(axis=1)]
# axis=1 => to consider at least one column value is missing

data2 = data.dropna(axis=0)

# Relationship between independent variables
correlation = data2.corr()

# Cross tables and data visualization 
# extracting the column names
data2.columns

# gender proportion table
gender = pd.crosstab(index = data2["gender"],
                     columns = 'count',
                     normalize = True)
print(gender)

# gender vs salary status
gender_salstat = pd.crosstab(index = data2["gender"],
                             columns = data2 ['SalStat'],
                             margins = True,
                             normalize = 'index')
print (gender_salstat)

# frequency distribution of 'salary status'
SalStat = sns.countplot(data2['SalStat'])

# histogram of age
sns.distplot(data2['age'], bins=10, kde = False)

# box plot - age vs salary status
sns.boxplot('SalStat', 'age', data=data2)
data2.groupby('SalStat')['age'].median()

# LOGISTIC REGRESSION

# reindexing the salary status names to 0, 1
data2['SalStat'] = data2['SalStat'].map({ ' less than or equal to 50,000':0, ' greater than 50,000':1})
print(data2['SalStat'])

new_data = pd.get_dummies(data2, drop_first = True)

# storing the column names
columns_list = list (new_data.columns)
print(columns_list)

# separating the input names from data
features = list(set(columns_list) -set (['SalStat']))
print(features)

# storing the output values in y
y = new_data['SalStat'].values
print(y)

# storing the values from input features
x = new_data[features].values
print(x)

# splitting the data into train and test 
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state =0)

# make an instance of the model
logistic = LogisticRegression()

# fitting the values for x and y
logistic.fit(train_x, train_y)
logistic.coef_
logistic.intercept_

# prediction from test data
prediction1 = logistic.predict(test_x)
print(prediction1)

# confusion matrix
confusion_matrix = confusion_matrix(test_y,prediction1)
print(confusion_matrix)

# calculating accuracy 
accuracy_score = accuracy_score(test_y, prediction1)
print(accuracy_score)

# printing the misclassified values from prediction
print('Misclassified samples: %d' % (test_y != prediction1). sum())

# logistic regression - removing insignificant variables
# reindexing the salary status names to 0,1
print(data2['SalStat'])

cols = ['gender', 'nativecountry', 'race', 'JobType']
new_data = data2.drop(cols,axis = 1)
new_data = pd.get_dummies(new_data, drop_first = True)

# storing the column names
columns_list = list(new_data.columns)
print(columns_list)

# separating the input names from data
features = list(set(columns_list) -set (['SalStat']))
print(features)

# storing the output values in y
y =  new_data['SalStat'].values
print(y)

# storing the values from input feaatures
x = new_data[features].values
print(x)

# splitting the data into train and test 
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state =0)

# make an instance of the model
logistic = LogisticRegression()
 
# fitting the values for x and y
logistic.fit(train_x, train_y)

# prediction from test data
prediction1 = logistic.predict(test_x)

# calculation accuracy
accuracy_score = accuracy_score(test_y, prediction1)
print(accuracy_score)
