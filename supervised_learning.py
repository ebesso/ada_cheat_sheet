import pandas as pd
import numpy as np

X = pd.DataFrame()
y = pd.DataFrame()
df = pd.DataFrame()

#################################################################################################################
# SPLIT AND PREPARE DATA 


# Fill in NaN values with mean
X = X.fillna(X.mean())

# One-Hot Encodes
X = pd.get_dummies(df['feature1, feature2'])

# Splits into test and train
# https://scikit-learn.org/stable/api/sklearn.model_selection.html
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)


#################################################################################################################
# LINEAR REGRESSION, LOGISTIC REGRESSION AND RIDGE REGRESSION

# Creates, train
# https://scikit-learn.org/stable/api/sklearn.linear_model.html
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge

# Creates a model
model = LinearRegression() 
model = LogisticRegression(solver='lbfgs')
model = Ridge(alpha=6)

# Train model
model.fit(X_train, y_train) 
model.fit(X_train, y_train) 
model.fit(X_train, y_train) 

# Makes prediction on trained lin_reg model
predictions = model.predict(X_test)

# Instead of splitting data, we can make predictions using cross-validation, then just insert an unfitted 
# model lin_reg
# https://scikit-learn.org/stable/api/sklearn.model_selection.html
from sklearn.model_selection import cross_val_predict
y_pred = cross_val_predict(model, X, y, cv=5)

##################################################################################################################
# VALIDATION


# TPR, FPR, Area Under Curve and Mean-Squared Error
# https://scikit-learn.org/stable/api/sklearn.metrics.html
from sklearn.metrics import mean_squared_error, auc, roc_curve, mean_absolute_error
mse = mean_squared_error(y, y_pred)
mae = mean_absolute_error(y, y_pred)
fpr, tpr, _ = roc_curve(y, y_pred)
auc_score = auc(fpr, tpr) # The higher the better, in case of ROC-plot

# Precision and Recall
from sklearn.model_selection import cross_val_score
precision = cross_val_score(model, X, y, cv=10, scoring="precision") # Precision: TP / (TP + FP)
recall = cross_val_score(model, X, y, cv=10, scoring="recall") # Recall: TP / (TP + FN)

#################################################################################################################
# K-NEAREST NEIGHBOR

# K-NN
from sklearn.neighbors import KNeighborsClassifier
# https://scikit-learn.org/stable/api/sklearn.neighbors.html#module-sklearn.neighbors
KNN15 = KNeighborsClassifier(15)
KNN15.fit(X_train, y_train)
KNN15.predict(X_test)

# See supervised_learning exercise for a plotting helper function 







