import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score

#######################################################
# load dataset
url = './DataExport/clean_dataset_9.csv'
table = pd.read_csv(url)
# fill the NaN values
table = table.fillna(0)

table['loyalty1'] = table['loggedIn']*table['isExclusiveMember']
table['loyalty2'] = table['loggedIn']*table['p_AddToCart']
table['loyalty3'] = table['loggedIn']*table['p_MapInteraction']
table['loyalty4'] = table['isExclusiveMember']*table['p_AddToCart']
table['loyalty5'] = table['p_sessionActivity']*table['p_TotalPrice']


#######################################################
# data exploration
print '******************DATA EXPLORATION*********************'
# Characteristics of Purchase and Not Purchase
print 'Booking Purchase Characteristic'
table_BP = table.groupby('BookingPurchase').mean()
print table_BP
# Purchase Behavior of People using different ostypes
print 'Booking Purchase Based on osType'
table_osType = table.groupby('osType').mean()
print table_osType
# Purchase Behavior of People of Different Gender
print 'Booking Purchase Based on Gender'
table_gender = table.groupby('gender').mean()
print table_gender

#######################################################
print '******************PREPARE DATA*************************'
# Prepare data for Logistic Regression
y, X = dmatrices('BookingPurchase ~  p_sessionActivity + C(p_AddToCart) + \
                  +p_sessionDuration + \
                  +C(isExclusiveMember) + C(loggedIn) + \
                   C(p_trafficChannel) + C(osType) + C(gender) \
                  + C(loyalty1) + C(loyalty2) + C(loyalty3) + C(loyalty4) + loyalty5', table, return_type='dataframe')

print X.columns


# flatten y into a 1-D array for regression
y = np.ravel(y)

########################################################
print '****************LOGISTIC REGRESSION*************************'
# instantiate a regression model, and fit with X and y
model = LogisticRegression()
#model = LogisticRegression(solver = 'newton-cg',max_iter = 3000, tol=0.00001)
model = model.fit(X, y)
# print the test score for my model
test_score = model.score(X, y)
print 'Model Accuracy: '
print test_score
print 'No Purchase Percentage:'
print 1-y.mean()

print '****************PRINT MODEL COEFFICIENTS*************************'
# examine the coefficients
print pd.DataFrame(zip(X.columns, np.transpose(model.coef_)))

#########################################################
print '*****************VALIDATION SET EVALUATION***********************'
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
model2 = LogisticRegression(solver = 'newton-cg',max_iter = 3000, tol=0.00001)
model2.fit(X_train, y_train)

# predict class labels for the test set
print 'Prediction on the Test Set'
predicted = model2.predict(X_test)
print predicted

# generate probabilities
print 'Probability on the Test Set'
probs = model2.predict_proba(X_test)
print probs

# test Accuracy
print 'Accuracy test using builtin functions'
print metrics.accuracy_score(y_test, predicted)
# area under the curve (AUC) score
print metrics.roc_auc_score(y_test, probs[:, 1])

