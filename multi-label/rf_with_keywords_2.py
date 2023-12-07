import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    hamming_loss,
)
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from imblearn.over_sampling import RandomOverSampler


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


# Load the data (update file paths as needed)
bug_data = load_data("data/processed/Bug.json")
discussion_data = load_data("data/processed/Discussion.json")
enhancement_data = load_data("data/processed/Enhancement.json")
feature_request_data = load_data("data/processed/Feature Request.json")
question_data = load_data("data/processed/Question.json")

# Combining all data for preprocessing
all_data = (
    bug_data + discussion_data + enhancement_data + feature_request_data + question_data
)

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

# Applying keyword weighting
weighted_texts = [
    weight_keywords(text, keywords[label[0]]) for text, label in zip(texts, labels)
]

# MultiLabel Binarizer
mlb = MultiLabelBinarizer()
encoded_labels = mlb.fit_transform(labels)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    weighted_texts, encoded_labels, test_size=0.2, random_state=42
)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Oversampling minority classes
ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train_tfidf, y_train)

# OneVsRestClassifier with RandomForestClassifier

ovr_classifier = OneVsRestClassifier(
    RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"),
)
ovr_classifier.fit(X_train_resampled, y_train_resampled)

# Predictions
y_pred = ovr_classifier.predict(X_test_tfidf)

# Evaluating the model
hamming_loss_score = hamming_loss(y_test, y_pred)
accuracy_score = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")

print("Hamming Loss:", hamming_loss_score)
print("Accuracy Score:", accuracy_score)
print("F1 Score:", f1)
print(
    "Classification Report:\n",
    classification_report(y_test, y_pred, target_names=mlb.classes_),
)
# print("Classification Report:\n", classification_report(y_test, y_pred))
