import pandas as pd
import numpy as np

"""
from zipfile import ZipFile

with ZipFile("nlp-getting-started.zip","r") as zipObj:
    zipObj.extractall()
    
"""
# Load data as pandas data frame & explore the fields
train_df = pd.read_csv('train.csv')
# Display 1st 10 rows
print(train_df.head(10))


def remove_special_char(text):
    n = len(text)
    formatted_text = []
    for i in range(0, n):
        # Remove special characters except space from the texts in each tweet
        formatted_text.append(''.join(c for c in text[i] if (c.isalnum() or c.isspace())))
    return formatted_text


train_df[["text"]] = train_df[["text"]].apply(remove_special_char)

print(train_df.head(10))
