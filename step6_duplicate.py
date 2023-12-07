import json
import os

# List of file paths to process
file_paths = [
    "data/processed/Bug.json",
    "data/processed/Question.json",
    "data/processed/Feature Request.json",
    "data/processed/Enhancement.json",
    "data/processed/Discussion.json",
]


# remove duplicate items based on issue_id in those files, and remove all of the items that are duplicated in the other files


def remove_duplicate():
    for file_path in file_paths:
        with open(file_path, "r") as file:
            data = json.load(file)
            for item in data:
                for file_path2 in file_paths:
                    if file_path2 != file_path:
                        with open(file_path2, "r") as file2:
                            data2 = json.load(file2)
                            for item2 in data2:
                                if item["issue_id"] == item2["issue_id"]:
                                    data.remove(item)
                                    break

            # Create the target directory if it doesn't exist
            target_directory = os.path.dirname(
                file_path.replace("processed", "duplicate")
            )
            if not os.path.exists(target_directory):
                os.makedirs(target_directory)

            output_file_path = file_path.replace("processed", "duplicate")
            with open(output_file_path, "w") as file:
                file.write(json.dumps(data, indent=4))


remove_duplicate()
