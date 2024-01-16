# Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Import the dataset
dataset = pd.read_csv("sample_data/50_Startups.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Encoding Categorical Data
ct = ColumnTransformer(transformers=[("enconder", OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Splitting dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)