import json
import os
import re


def separate_code_and_words(description):
    """
    Separates the code (enclosed in ```) and human words from a given string.
    Returns two lists: one containing code snippets and the other containing human words.
    """
    # Regular expression to find code snippets enclosed in triple backticks
    code_pattern = r"```(.*?)```"
    code_snippets = re.findall(code_pattern, description, re.DOTALL)

    # Remove code snippets from the description
    words_only = re.sub(code_pattern, "", description)

    return code_snippets, words_only


def process_json_file(file_path):
    """
    Processes a JSON file to extract and separate code and human words from
    the 'remove_template_description' key, if it exists.
    Returns the data with new keys of 'codes' and 'human_words'.
    """
    with open(file_path, "r") as file:
        data = json.load(file)

    for item in data:
        if "remove_template_description" in item:
            codes, human_words = separate_code_and_words(
                item["remove_template_description"]
            )
            item["codes_and_errors"] = codes
            item["human_words"] = human_words

    return data


# Example usage
file_paths = [
    "data/remove_template/Bug.json",
    "data/remove_template/Question.json",
    "data/remove_template/Feature Request.json",
    "data/remove_template/Enhancement.json",
    "data/remove_template/Discussion.json",
]

# Save the results to a JSON file in data/separated
for file_path in file_paths:
    separated_data = process_json_file(file_path)
    filename = os.path.basename(file_path)
    # Create the target directory if it doesn't exist
    target_directory = os.path.dirname(
        file_path.replace("remove_template", "separated")
    )
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
    with open(f"{target_directory}/{filename}", "w") as f:
        f.write(json.dumps(separated_data, indent=4))
