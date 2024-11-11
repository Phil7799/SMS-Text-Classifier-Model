# import libraries
import tensorflow as tf
import pandas as pd
from tensorflow import keras
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import requests

print(tf.__version__)

# Download the data files from URLs
train_url = "https://cdn.freecodecamp.org/project-data/sms/train-data.tsv"
test_url = "https://cdn.freecodecamp.org/project-data/sms/valid-data.tsv"

train_file_path = "train-data.tsv"
test_file_path = "valid-data.tsv"

# Download the files
train_response = requests.get(train_url)
test_response = requests.get(test_url)

# Save the files locally
with open(train_file_path, 'wb') as file:
    file.write(train_response.content)

with open(test_file_path, 'wb') as file:
    file.write(test_response.content)

# Load the data from the saved local files
train_data = pd.read_csv(train_file_path, sep='\t', header=None, names=['label', 'message'])
test_data = pd.read_csv(test_file_path, sep='\t', header=None, names=['label', 'message'])

# Encoding labels: "ham" -> 0, "spam" -> 1
train_data['label'] = train_data['label'].map({'ham': 0, 'spam': 1})
test_data['label'] = test_data['label'].map({'ham': 0, 'spam': 1})

# Tokenizing the text messages
tokenizer = keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(train_data['message'])

# Converting text messages to sequences
X_train = tokenizer.texts_to_sequences(train_data['message'])
X_test = tokenizer.texts_to_sequences(test_data['message'])

# Padding sequences for uniform input length
X_train = keras.preprocessing.sequence.pad_sequences(X_train, padding='post')
X_test = keras.preprocessing.sequence.pad_sequences(X_test, padding='post')

# Preparing labels
y_train = train_data['label']
y_test = test_data['label']

# Building the model
model = keras.Sequential([
    keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64, input_length=X_train.shape[1]),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model for 20 epochs
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

# Save the model
model.save('sms_classifier_model.h5')

# Define the predict_message function
def predict_message(message, threshold=0.5):
    # Convert the input message to a sequence
    seq = tokenizer.texts_to_sequences([message])
    padded = keras.preprocessing.sequence.pad_sequences(seq, padding='post', maxlen=X_train.shape[1])

    # Make prediction
    prediction = model.predict(padded)
    
    # Print the prediction for debugging
    print(f"Prediction for '{message}': {prediction[0][0]}")

    # Return the likeliness (probability) and the classification
    prob = prediction[0][0]
    label = "spam" if prob > threshold else "ham"
    return [prob, label]

# Testing the function with 1st 2 different examples
print(predict_message("Free money now! Call 123456"))
print(predict_message("Hey, how are you doing today?"))

# Run this cell to test your function and model
def test_predictions():
    test_messages = [
        "how are you doing today",
        "sale today! to stop texts call 98912460324",
        "i dont want to go. can we try it a different day? available sat",
        "our new mobile video service is live. just install on your phone to start watching.",
        "you have won £1000 cash! call to claim your prize.",
        "i'll bring it tomorrow. don't forget the milk.",
        "wow, is your arm alright. that happened to me one time too"
    ]

    test_answers = ["ham", "spam", "ham", "spam", "spam", "ham", "ham"]
    passed = True

    for msg, ans in zip(test_messages, test_answers):
        prediction = predict_message(msg)
        if prediction[1] != ans:
            passed = False

    if passed:
        print("You passed the challenge. Great job!")
    else:
        print("You haven't passed yet. Keep trying.")

test_predictions()