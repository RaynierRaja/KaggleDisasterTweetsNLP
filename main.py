import pandas as pd
import numpy as np
import tensorflow as tf

""" 
1. Load the train.csv as pandas data frame
2. explore and preprocess it
"""
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

for tweet, label in tf_dataset.take(1):
    print('tweet: ', tweet.numpy())
    print('label: ', label.numpy())

DATASET_SIZE = len(tf_dataset)
TRAIN_SIZE = int(0.8 * DATASET_SIZE)
TEST_SIZE = int(0.2 * DATASET_SIZE)
# Data set for training
test_tf_dataset = tf_dataset.take(TEST_SIZE)
train_tf_dataset = tf_dataset.skip(TEST_SIZE)

# Text encoder
VOCAB_SIZE = 1000
encoder = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE)
encoder.adapt(train_tf_dataset.map(lambda text, tgt: text))

print(tweet)
print(encoder(tweet))

# Prepare the model
model = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(input_dim=len(encoder.get_vocabulary()), output_dim=128, mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1)
])

print([layer.supports_masking for layer in model.layers])

# Check if the model is working
sample_tweet = tweet.numpy()
predictions = model.predict(np.array([sample_tweet]))
print(predictions)


"""Train the model"""

# Shuffle dataset
BUFFER_SIZE = 10000
BATCH_SIZE = 64
train_tf_dataset = train_tf_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_tf_dataset = test_tf_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
# Compile model
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])
# Train the model
history = model.fit(train_tf_dataset, epochs=10,
                    validation_data=test_tf_dataset,
                    validation_steps=30)

test_loss, test_acc = model.evaluate(test_tf_dataset)

print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)