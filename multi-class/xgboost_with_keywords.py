import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE


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

# Applying keyword weighting
weighted_texts = [
    weight_keywords(text, keywords[label]) for text, label in zip(texts, labels)
]


# Label encoding
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    weighted_texts, encoded_labels, test_size=0.2, random_state=42
)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Applying SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_tfidf, y_train)

# Train the model with SMOTE applied data using Gradient Boosting
gb_classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_classifier.fit(X_train_smote, y_train_smote)

# Predictions
y_pred = gb_classifier.predict(X_test_tfidf)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("\nClassification Report:\n", report)
