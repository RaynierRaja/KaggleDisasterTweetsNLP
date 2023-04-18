import pandas as pd
import numpy as np

"""
from zipfile import ZipFile

with ZipFile("nlp-getting-started.zip","r") as zipObj:
    zipObj.extractall()
    
"""
# Load data as pandas data frame & explore the fields
train_df = pd.read_csv('train.csv')
# 1st 10 rows
print(train_df.head(10))

