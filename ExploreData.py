import pandas as pd
import tensorflow as tf

"""
# Extract the ZIP file to get train and test data
from zipfile import ZipFile

with ZipFile("nlp-getting-started.zip","r") as zipObj:
    zipObj.extractall()
"""

""" 
1. Load the train.csv as pandas data frame
2. explore and preprocess it
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

print(train_df["text"][1])


""" 
1. Prepare the dataset for training
"""
# Convert Pandas Data Frame to TF Dataset
feature = train_df.pop('text')
target = train_df.pop('target')
tf_dataset = tf.data.Dataset.from_tensor_slices((feature, target))

for tweet, label in tf_dataset.take(1):
    print('text: ', tweet.numpy())
    print('label: ', label.numpy())

DATASET_SIZE = len(tf_dataset)
TRAIN_SIZE = int(0.8 * DATASET_SIZE)
TEST_SIZE = int(0.2 * DATASET_SIZE)

test_tf_dataset = tf_dataset.take(TEST_SIZE)
train_tf_dataset = tf_dataset.skip(TEST_SIZE)

