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
url = './DataExport/clean_dataset_1.csv'
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
# Purchase Behavior of People of Different Gender
print 'Booking Purchase Based on Gender'
table_gender = table.groupby('gender').mean()
print table_gender

#######################################################
print '******************PREPARE DATA*************************'
# Prepare data for Logistic Regression
y, X = dmatrices('BookingPurchase ~ gender + p_sessionActivity + p_AddToCart + \
                  +p_sessionDuration + p_pageViews + daysFromPreviousVisit + \
                  +C(isExclusiveMember) + C(loggedIn) + C(p_MapInteraction) + \
                  C(p_trafficChannel) + C(osType)', table, return_type='dataframe')
print X.columns
"""
Bug note: No columns for u'C(isExclusiveMember)[T.0]', u'C(loggedIn)[T.0]',
       u'C(p_MapInteraction)[T.0]' We might have to change variable types for
       those columns for things to work properly. But for now, I'll leave the bug
       there when I'm testing
"""
# clean up the category names in X for readability
#X = X.rename(columns = {'C(occupation)[T.2.0]':'occ_2',
#                        'C(occupation)[T.3.0]':'occ_3',
#                        'C(occupation)[T.4.0]':'occ_4',
#                        'C(occupation)[T.5.0]':'occ_5',
#                        'C(occupation)[T.6.0]':'occ_6',
#                        'C(occupation_husb)[T.2.0]':'occ_husb_2',
#                        'C(occupation_husb)[T.3.0]':'occ_husb_3',
#                        'C(occupation_husb)[T.4.0]':'occ_husb_4',
#                        'C(occupation_husb)[T.5.0]':'occ_husb_5',
#                        'C(occupation_husb)[T.6.0]':'occ_husb_6'})

# flatten y into a 1-D array for regression
y = np.ravel(y)

########################################################
print '****************LOGISTIC REGRESSION*************************'
# instantiate a regression model, and fit with X and y
model = LogisticRegression()
model = model.fit(X, y)
# print the test score for my model
test_score = model.score(X, y)
print test_score
