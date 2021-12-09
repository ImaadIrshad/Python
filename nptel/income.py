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
