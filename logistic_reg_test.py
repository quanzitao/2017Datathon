import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

from patsy import dmatrices
#from sklearn.linear_model import LogisticRegression
#from sklearn.cross_validation import train_test_split
#from sklearn import metrics
#from sklearn.cross_validation import cross_val_score

#######################################################
# load dataset
url = './DataExport/clean_dataset_1.csv'
table = pd.read_csv(url)

# data exploration
table = table.groupby('BookingPurchase').mean()
print table
