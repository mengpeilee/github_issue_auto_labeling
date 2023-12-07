import json
import os
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import download
from nltk.stem import WordNetLemmatizer

# Downloading necessary NLTK data
download("punkt")
download("stopwords")
download("wordnet")

# Initializing NLTK tools
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def process_text(text):
    # make text lowercase
    text = text.lower()
    # Tokenizing the text
    tokens = word_tokenize(text)

    # Removing stopwords
    no_stopwords = [word for word in tokens if word not in stop_words]

    # Stemming
    stemmed = [stemmer.stem(word) for word in tokens]

    # Lemmatization
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens]

    # Removing stopwords and then lemmatizing
    no_stopwords_lemmatized = [lemmatizer.lemmatize(word) for word in no_stopwords]

    return {
        "stopwords_removal": " ".join(no_stopwords),
        "stemmed": " ".join(stemmed),
        "lemmatized_description": " ".join(lemmatized),
        "stopwords_removal_lemmatization": " ".join(no_stopwords_lemmatized),
    }


def process_json_file(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)

    for item in data:
        if "human_words_regrex" in item:
            # remove item from data if  "human_words_regrex" is null
            if item["human_words_regrex"] is None:
                data.remove(item)
                continue

            # remove "codes_and_errors" from data
            # if "codes_and_errors" in item:
            #     del item["codes_and_errors"]

            processed_text = process_text(item["human_words_regrex"])
            processed_title = process_text(item["title"])
            item["human_words_stopwords_removal"] = processed_text["stopwords_removal"]
            item["human_words_stemmed"] = processed_text["stemmed"]
            item["human_words_lemmatized_description"] = processed_text["lemmatized_description"]
            item["human_words_stopwords_removal_lemmatization"] = processed_text[
                "stopwords_removal_lemmatization"
            ]
            item["title_stopwords_removal_lemmatization"] = processed_title[
                "stopwords_removal_lemmatization"
            ]
    return data


# Processing the 'human_words_regrex' field for each record in each file

# List of file paths to process
file_paths = [
    "data/regrex/Bug.json",
    "data/regrex/Question.json",
    "data/regrex/Feature Request.json",
    "data/regrex/Enhancement.json",
    "data/regrex/Discussion.json",
]

for file_path in file_paths:
    processed_data = process_json_file(file_path)

    # Create the target directory if it doesn't exist
    target_directory = os.path.dirname(file_path.replace("regrex", "processed"))
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    output_file_path = file_path.replace("regrex", "processed")

    with open(output_file_path, "w") as f:
        f.write(json.dumps(processed_data, indent=4))
