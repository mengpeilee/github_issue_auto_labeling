import json
import os


# Function to clean up the description
def clean_description(data):
    # Remove specific keys
    keys_to_remove = [
        "### Website or app",
        "### Repro steps",
        "### How often does this bug happen?",
        "### DevTools package (automated)",
        "### DevTools version (automated)",
        "### Error message (automated)",
        "### Error call stack (automated)",
        "### Error component stack (automated)",
        "### GitHub query string (automated)",
        "No response",
        "## Steps To Reproduce",
        "## The current behavior",
        "## The expected behavior",
    ]

    # Retrieve the 'description' field, handling None values
    description = data.get("description", "") or ""
    # Replace \n and \r with white space in the description
    description = description.replace("\n", " ").replace("\r", " ")
    # Remove the specified keys from the description
    for key in keys_to_remove:
        description = description.replace(key, "")

    return description.strip()


# Function to handle list of dictionaries format
def clean_descriptions_in_list(data_list):
    cleaned_data_list = []
    for data in data_list:
        # if "description": null, drpo the data
        if data["description"] is None:
            continue
        cleaned_description = clean_description(data)
        data["remove_template_description"] = cleaned_description
        cleaned_data_list.append(data)
    return cleaned_data_list


# List of file paths to process
file_paths = [
    "data/original/Bug.json",
    "data/original/Question.json",
    "data/original/Feature Request.json",
    "data/original/Enhancement.json",
    "data/original/Discussion.json",
]

for file_path in file_paths:
    with open(file_path, "r") as file:
        data = json.load(file)
    cleaned_data = clean_descriptions_in_list(data)

    # Create the target directory if it doesn't exist
    target_directory = os.path.dirname(file_path.replace("original", "remove_template"))
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    output_file_path = file_path.replace("original", "remove_template")

    with open(output_file_path, "w") as f:
        f.write(json.dumps(cleaned_data, indent=4))

# Load the file and clean the descriptions
# file_path = '/mnt/data/Bug.json'
# with open(file_path, 'r', encoding='utf-8') as file:
#     bug_data = json.load(file)
# cleaned_bug_data = clean_descriptions_in_list(bug_data)
