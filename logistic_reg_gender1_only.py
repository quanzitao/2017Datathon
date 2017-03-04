import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score

#######################################################
# load dataset
url = './DataExport/dataset_gender1_only.csv'
table = pd.read_csv(url)

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

#######################################################
print '******************PREPARE DATA*************************'
# Prepare data for Logistic Regression
#y, X = dmatrices('BookingPurchase ~  p_sessionActivity + p_AddToCart + \
#                  +p_sessionDuration + p_pageViews + daysFromPreviousVisit + \
#                  +C(isExclusiveMember) + C(loggedIn) + C(p_MapInteraction) + \
#                  C(p_trafficChannel) + C(osType) + C(gender)', table, return_type='dataframe')

y, X = dmatrices('BookingPurchase ~ p_sessionActivity + p_AddToCart + \
                  +C(isExclusiveMember) + C(loggedIn) + \
                  C(p_trafficChannel) + C(osType)', table, return_type='dataframe')

print X.columns

"""
Bug note: No columns for u'C(isExclusiveMember)[T.0]', u'C(loggedIn)[T.0]',
       u'C(p_MapInteraction)[T.0]' We might have to change variable types for
       those columns for things to work properly. But for now, I'll leave the bug
       there when I'm testing
"""

# flatten y into a 1-D array for regression
y = np.ravel(y)

########################################################
print '****************LOGISTIC REGRESSION*************************'
# instantiate a regression model, and fit with X and y
model = LogisticRegression()
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
model2 = LogisticRegression()
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

