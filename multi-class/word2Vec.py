import json
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from nltk.tokenize import word_tokenize
import nltk

nltk.download("punkt")


# Function to load data
def load_data(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


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

# Tokenize texts for Word2Vec
tokenized_texts = [word_tokenize(text.lower()) for text in texts]

# Train or load a Word2Vec model
word2vec_model = Word2Vec(
    tokenized_texts, vector_size=5000, window=5, min_count=1, workers=4
)


# Function to create averaged word vector for a text
def average_word_vectors(words, model, vocabulary, num_features):
    feature_vector = np.zeros((num_features,), dtype="float32")
    nwords = 0

    for word in words:
        if word in vocabulary:
            nwords = nwords + 1
            feature_vector = np.add(feature_vector, model.wv[word])

    if nwords:
        feature_vector = np.divide(feature_vector, nwords)

    return feature_vector


# Averaging the Word2Vec embeddings for each text
vocab = set(word2vec_model.wv.index_to_key)
features = [
    average_word_vectors(tokenized_text, word2vec_model, vocab, 5000)
    for tokenized_text in tokenized_texts
]

# Label encoding
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    features, encoded_labels, test_size=0.2, random_state=42
)

# Applying SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(np.array(X_train), y_train)

# # Train the model with SMOTE applied data using Gradient Boosting
# gb_classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)
# gb_classifier.fit(X_train_smote, y_train_smote)

# # Predictions
# y_pred = gb_classifier.predict(np.array(X_test))

# # Evaluating the model
# accuracy = accuracy_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred, average="weighted")
# report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

# print("Accuracy:", accuracy)
# print("F1 Score:", f1)
# print("\nClassification Report:\n", report)


# Train the model with SMOTE applied data
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_smote, y_train_smote)

# Predictions
y_pred = rf_classifier.predict(np.array(X_test))

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("\nClassification Report:\n", report)
