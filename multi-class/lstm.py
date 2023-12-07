import json
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, SpatialDropout1D
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, accuracy_score, f1_score


# Function to load data
def load_data(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


# Keywords identified for each category
keywords = {
    "Bug": [
        "react",
        "component",
        "error",
        "update",
        "state",
        "render",
        "behavior",
        "current",
        "hook",
        "issue",
    ],
    "Discussion": [
        "error",
        "run",
        "possible",
        "useeffect",
        "callback",
        "library",
        "demo",
        "input",
        "listener",
        "effect",
    ],
    "Enhancement": [
        "devtools",
        "warning",
        "dependency",
        "expected",
        "currently",
        "variable",
        "link",
        "handle",
        "app",
        "tree",
    ],
    "Feature Request": [
        "problem",
        "solution",
        "ref",
        "behavior",
        "current",
        "return",
        "node",
        "context",
        "usememo",
        "argument",
    ],
    "Question": [
        "value",
        "change",
        "event",
        "setstate",
        "set",
        "want",
        "api",
        "console",
        "html",
        "help",
    ],
}


# Function to weight keywords in text
def weight_keywords(text, category_keywords, duplication_factor=3):
    for keyword in category_keywords:
        text = text.replace(keyword, (keyword + " ") * duplication_factor)
    return text


# Load the data
bug_data = load_data("data/duplicate/Bug.json")
discussion_data = load_data("data/duplicate/Discussion.json")
enhancement_data = load_data("data/duplicate/Enhancement.json")
feature_request_data = load_data("data/duplicate/Feature Request.json")
question_data = load_data("data/duplicate/Question.json")

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
    ["Bug"] * len(bug_data)
    + ["Discussion"] * len(discussion_data)
    + ["Enhancement"] * len(enhancement_data)
    + ["Feature Request"] * len(feature_request_data)
    + ["Question"] * len(question_data)
)

# Label encoding
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Applying keyword weighting
weighted_texts = [
    weight_keywords(text, keywords[label]) for text, label in zip(texts, labels)
]

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    weighted_texts, encoded_labels, test_size=0.2, random_state=42
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

# Applying SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_pad, y_train)

# One-hot encoding of labels
y_train_smote = to_categorical(y_train_smote)
y_test_categorical = to_categorical(y_test)

# LSTM Model
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=100, input_length=max_seq_length))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(len(label_encoder.classes_), activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model
model.fit(X_train_smote, y_train_smote, epochs=5, batch_size=64, validation_split=0.1)

# Predictions
y_pred = model.predict(X_test_pad)
y_pred_classes = np.argmax(y_pred, axis=1)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred_classes)
f1 = f1_score(y_test, y_pred_classes, average="weighted")
report = classification_report(
    y_test, y_pred_classes, target_names=label_encoder.classes_
)

print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("\nClassification Report:\n", report)
