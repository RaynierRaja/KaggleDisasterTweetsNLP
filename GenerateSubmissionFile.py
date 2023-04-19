import tensorflow as tf
import numpy as np
import pandas as pd

model = tf.keras.models.load_model('DisasterTweetClassifier')

# Load test data as pandas data frame & explore the fields
test_df = pd.read_csv('test.csv')


def remove_special_char(text):
    n = len(text)
    formatted_text = []
    for i in range(0, n):
        # Remove special characters except space from the texts in each tweet
        formatted_text.append(''.join(c for c in text[i] if (c.isalnum() or c.isspace())))
    return formatted_text


test_df[["text"]] = test_df[["text"]].apply(remove_special_char)
print(test_df["text"][1])

# Convert Pandas Data Frame to TF Dataset
feature = test_df.pop('text')
print(np.array(feature))
print(len(np.array(feature)))

target = model.predict(np.array(feature))

# Load submission file
sub_df = pd.read_csv('sample_submission.csv')
j = 0
for pred in target:
    sub_df.loc[j, 'target'] = 1 if pred > 0 else 0
    j = j + 1

print(sub_df.head(5))
sub_df.to_csv('submissions.csv', index=False)
