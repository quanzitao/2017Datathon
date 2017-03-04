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

#WHOLE DATASET export dataset for viewing
raw_table.head()
raw_table.to_csv('./DataExport/clean_dataset_1.csv')

#GENDER 1 ONLY DATASET export dataset
table_gender1_only = raw_table
table_gender1_only = table_gender1_only.drop(table_gender1_only[table_gender1_only.gender == 0].index)
table_gender1_only.head()
table_gender1_only.to_csv('./DataExport/dataset_gender1_only.csv')

#GENDER 0 ONLY DATASET export dataset
table_gender0_only = raw_table
table_gender0_only = table_gender1_only.drop(table_gender0_only[table_gender0_only.gender == 1].index)
table_gender0_only.head()
table_gender0_only.to_csv('./DataExport/dataset_gender0_only.csv')

#isExclusiveMember 1 ONLY DATASET export dataset
table_isEM1_only = raw_table
table_isEM1_only = table_isEM1_only.drop(table_isEM1_only[table_isEM1_only.isExclusiveMember == 0].index)
table_isEM1_only.head()
table_isEM1_only.to_csv('./DataExport/dataset_isEM1_only.csv')

#isExclusiveMember 0 ONLY DATASET export dataset
table_isEM0_only = raw_table
table_isEM0_only = table_isEM0_only.drop(table_isEM0_only[table_isEM0_only.isExclusiveMember == 1].index)
table_isEM0_only.head()
table_isEM0_only.to_csv('./DataExport/dataset_isEM0_only.csv')
                        
