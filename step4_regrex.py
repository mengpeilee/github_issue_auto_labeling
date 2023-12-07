import json
import re
import os


def process_file(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)

    for item in data:
        description = item["human_words"]

        if description:
            # If space is more than 1, replace with 1 space
            processed_description = re.sub(r"\s+", " ", description)

            # Remove Markdown link syntax and URLs
            processed_description = re.sub(
                r"\[.*?\]\(.*?\)", "", processed_description
            )  # Remove Markdown links
            processed_description = re.sub(
                r"http\S+", "", processed_description
            )  # Remove URLs
            processed_description = re.sub(
                r"https?://\S+", "", processed_description
            )  # Remove URLs with "://"
            processed_description = re.sub(
                r"<[^>]+>", "", processed_description
            )  # Remove HTML tags

            # Add the processed description to the item
            item["human_words_regrex"] = processed_description
        else:
            # Set description_regrex to None if description is None
            item["human_words_regrex"] = None

    return data


# List of file paths to process
file_paths = [
    "data/separated/Bug.json",
    "data/separated/Question.json",
    "data/separated/Feature Request.json",
    "data/separated/Enhancement.json",
    "data/separated/Discussion.json",
]

for file_path in file_paths:
    processed_data = process_file(file_path)

    # Create the target directory if it doesn't exist
    target_directory = os.path.dirname(file_path.replace("separated", "regrex"))
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    output_file_path = file_path.replace("separated", "regrex")

    with open(output_file_path, "w") as f:
        f.write(json.dumps(processed_data, indent=4))
