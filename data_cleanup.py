import pandas as pd
import numpy as np

url = './TripAdvisorChallenge/datathon_tadata.csv'
raw_table = pd.read_csv(url)

#clean up the NA values
raw_table = pd.DataFrame(raw_table)
pd.to_numeric(raw_table.daysToCheckin)
pd.to_numeric(raw_table.p_TotalPrice)

#replace NA with negative value
raw_table.fillna(0)

#export dataset for viewing
raw_table.head()
print raw_table
raw_table.to_csv('./Data_Export/clean_dataset_1.csv')
