import tensorflow as tf
import numpy as np
import pandas as pd

model = tf.keras.models.load_model('DisasterTweetClassifier')

# Load data as pandas data frame & explore the fields
train_df = pd.read_csv('train.csv')


def remove_special_char(text):
    n = len(text)
    formatted_text = []
    for i in range(0, n):
        # Remove special characters except space from the texts in each tweet
        formatted_text.append(''.join(c for c in text[i] if (c.isalnum() or c.isspace())))
    return formatted_text


train_df[["text"]] = train_df[["text"]].apply(remove_special_char)
print(train_df["text"][1])

""" 
1. Prepare the dataset for training
"""
# Convert Pandas Data Frame to TF Dataset
feature = train_df.pop('text')
target = train_df.pop('target')
tf_dataset = tf.data.Dataset.from_tensor_slices((feature, target))

DATASET_SIZE = len(tf_dataset)
TRAIN_SIZE = int(0.8 * DATASET_SIZE)
TEST_SIZE = int(0.2 * DATASET_SIZE)
# Data set for training
test_tf_dataset = tf_dataset.take(TEST_SIZE)
train_tf_dataset = tf_dataset.skip(TEST_SIZE)

BUFFER_SIZE = 1000
BATCH_SIZE = 64
train_tf_dataset = train_tf_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_tf_dataset = test_tf_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_loss, test_acc = model.evaluate(test_tf_dataset)

print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)

