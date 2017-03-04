import pandas as pd
import numpy as np

url = './TripAdvisorChallenge/datathon_tadata.csv'
raw_table = pd.read_csv(url)

#clean up the NA values
raw_table = pd.DataFrame(raw_table)
pd.to_numeric(raw_table.daysToCheckin)
pd.to_numeric(raw_table.p_TotalPrice)

#delete osTypeName column
raw_table.drop('osTypeName', axis=1, inplace=True)

#export dataset for viewing
raw_table.head()
print raw_table
raw_table.to_csv('./DataExport/clean_dataset_1.csv')
