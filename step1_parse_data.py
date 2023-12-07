import requests
import json


def get_issues(owner, repo, label):
    issues = []
    page = 1
    while True:
        issues_url = f"https://api.github.com/repos/{owner}/{repo}/issues"
        params = {
            "state": "all",
            "labels": label,
            "per_page": 100,  # Maximum items per page
            "page": page,
        }
        response = requests.get(issues_url, params=params)
        page_issues = response.json()
        if not page_issues or "message" in page_issues:
            break
        issues.extend(page_issues)
        page += 1

    parsed_issues = []

    for issue in issues:
        if "pull_request" not in issue:  # to exclude pull requests
            parsed_issue = {
                "issue_id": issue["id"],
                "title": issue["title"],
                "label": [label["name"] for label in issue["labels"]],
                "date": issue["created_at"],
                "status": issue["state"],
                "description": issue["body"],
            }
            parsed_issues.append(parsed_issue)

    return json.dumps(parsed_issues, indent=4)


# Example usage
owner = "facebook"
repo = "react"
labels = [
    "Type: Bug",
    "Type: Question",
    "Type: Feature Request",
    "Type: Enhancement",
    "Type: Discussion",
]

for label in labels:
    issues_json = get_issues(owner, repo, label)
    filename = label.replace("Type: ", "")
    with open(f"data/original/{filename}.json", "w") as f:
        f.write(issues_json)
