import json
import gensim
from gensim.parsing.preprocessing import STOPWORDS
from gensim.utils import simple_preprocess
from collections import defaultdict


# Function to load data
def load_data(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


# Function to preprocess text: stopword removal and lemmatization
def preprocess_text(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in STOPWORDS:
            result.append(token)
    return result


# Function to handle potential exceptions during preprocessing
def preprocess_with_exception_handling(data):
    processed_texts = []
    for entry in data:
        try:
            processed_texts.append(
                preprocess_text(entry["human_words_stopwords_removal_lemmatization"])
            )
        except KeyError:
            continue  # Skip entries with missing keys
    return processed_texts


# Function to calculate frequency of each word
def calculate_frequency(texts):
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1
    return frequency


# Load the data
bug_data = load_data("data/processed/Bug.json")
discussion_data = load_data("data/processed/Discussion.json")
enhancement_data = load_data("data/processed/Enhancement.json")
feature_request_data = load_data("data/processed/Feature Request.json")
question_data = load_data("data/processed/Question.json")


# Preprocessing the data
bug_texts = preprocess_with_exception_handling(bug_data)
discussion_texts = preprocess_with_exception_handling(discussion_data)
enhancement_texts = preprocess_with_exception_handling(enhancement_data)
feature_request_texts = preprocess_with_exception_handling(feature_request_data)
question_texts = preprocess_with_exception_handling(question_data)

# Calculating frequency for each category
bug_freq = calculate_frequency(bug_texts)
discussion_freq = calculate_frequency(discussion_texts)
enhancement_freq = calculate_frequency(enhancement_texts)
feature_request_freq = calculate_frequency(feature_request_texts)
question_freq = calculate_frequency(question_texts)

# Calculating the top 10 words for each category
bug_top10 = sorted(bug_freq.items(), key=lambda x: x[1], reverse=True)[:10]
discussion_top10 = sorted(discussion_freq.items(), key=lambda x: x[1], reverse=True)[
    :10
]
enhancement_top10 = sorted(enhancement_freq.items(), key=lambda x: x[1], reverse=True)[
    :10
]
feature_request_top10 = sorted(
    feature_request_freq.items(), key=lambda x: x[1], reverse=True
)[:10]
question_top10 = sorted(question_freq.items(), key=lambda x: x[1], reverse=True)[:10]

# Printing the top 10 words for each category
print("Bug: ", bug_top10)
print("Discussion: ", discussion_top10)
print("Enhancement: ", enhancement_top10)
print("Feature Request: ", feature_request_top10)
print("Question: ", question_top10)
