import json


# count the number of issues in each file
labels = [
    "Type: Bug",
    "Type: Question",
    "Type: Feature Request",
    "Type: Enhancement",
    "Type: Discussion",
]


def count_issues():
    for label in labels:
        filename = label.replace("Type: ", "")
        with open(f"data/processed/{filename}.json", "r") as f:
            data = json.load(f)
            print(f"{label}: {len(data)}")


count_issues()

# remove label that is not in the list in each json file
