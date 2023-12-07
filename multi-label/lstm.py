import json
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, SpatialDropout1D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
    hamming_loss,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    classification_report,
)


# Function to load data
def load_data(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


# Load the data
bug_data = load_data("data/processed/Bug.json")
discussion_data = load_data("data/processed/Discussion.json")
enhancement_data = load_data("data/processed/Enhancement.json")
feature_request_data = load_data("data/processed/Feature Request.json")
question_data = load_data("data/processed/Question.json")

# Combining all data for preprocessing
all_data = (
    bug_data + discussion_data + enhancement_data + feature_request_data + question_data
)

# Extracting text and labels
texts = [
    (
        d.get("title_stopwords_removal_lemmatization", "")
        + " "
        + d.get("human_words_stopwords_removal_lemmatization", "")
    ).strip()
    for d in all_data
]
labels = (
    [["Bug"] for _ in range(len(bug_data))]
    + [["Discussion"] for _ in range(len(discussion_data))]
    + [["Enhancement"] for _ in range(len(enhancement_data))]
    + [["Feature Request"] for _ in range(len(feature_request_data))]
    + [["Question"] for _ in range(len(question_data))]
)

# Multi-label binarization of labels
mlb = MultiLabelBinarizer()
encoded_labels = mlb.fit_transform(
    [set(label[0].split()) for label in labels]
)  # Adjust based on your label format

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    texts, encoded_labels, test_size=0.2, random_state=42
)

# Tokenization
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Padding sequences
max_seq_length = 100  # You might need to adjust this
X_train_pad = pad_sequences(X_train_seq, maxlen=max_seq_length)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_seq_length)

# LSTM Model
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=100, input_length=max_seq_length))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(len(mlb.classes_), activation="sigmoid"))  # sigmoid for multi-label

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model
model.fit(X_train_pad, y_train, epochs=5, batch_size=64, validation_split=0.1)

# Predictions
y_pred = model.predict(X_test_pad)
y_pred_classes = np.argmax(y_pred, axis=1)

# Convert probabilities to binary values
y_pred_binary = (y_pred > 0.5).astype(int)

# Evaluating the model with appropriate metrics
hammingLoss = hamming_loss(y_test, y_pred_binary)
precision = precision_score(y_test, y_pred_binary, average="weighted")
recall = recall_score(y_test, y_pred_binary, average="weighted")
accuracy_score = accuracy_score(y_test, y_pred_binary)
f1 = f1_score(y_test, y_pred_binary, average="weighted")

print("Hamming Loss:", hammingLoss)
print("Precision:", precision)
print("Recall:", recall)
print("Accuracy:", accuracy_score)
print("F1 Score:", f1)
print(
    "Classification Report:\n",
    classification_report(y_test, y_pred_binary, target_names=mlb.classes_),
)
