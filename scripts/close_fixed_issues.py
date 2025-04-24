import os
import re
import subprocess
from github import Github

def get_latest_commit_message():
    """Get the latest commit message."""
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--pretty=%B"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        print("Error getting commit message")
        return ""

def extract_issue_numbers(commit_message):
    """Extract issue numbers from commit message using GitHub's closing keywords."""
    # GitHub's closing keywords: close, closes, closed, fix, fixes, fixed, resolve, resolves, resolved
    pattern = r'(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s+#(\d+)'
    matches = re.findall(pattern, commit_message, re.IGNORECASE)
    return [int(issue_num) for issue_num in matches]

def close_issues(issue_numbers):
    """Close the specified issues on GitHub."""
    token = os.environ.get('GITHUB_TOKEN')
    if not token:
        raise ValueError("Missing GITHUB_TOKEN environment variable")

    g = Github(token)
    repo_name = os.environ.get('GITHUB_REPOSITORY', 'Lemniscate-world/Neural')
    repo = g.get_repo(repo_name)

    for issue_num in issue_numbers:
        try:
            issue = repo.get_issue(issue_num)
            if issue.state == 'open':
                # Close the issue
                issue.edit(state='closed')
                # Add a comment
                issue.create_comment("This issue was automatically closed by the CI system because it was fixed in a recent commit.")
                print(f"Closed issue #{issue_num}")
            else:
                print(f"Issue #{issue_num} is already closed")
        except Exception as e:
            print(f"Error closing issue #{issue_num}: {str(e)}")

if __name__ == "__main__":
    commit_message = get_latest_commit_message()
    print(f"Commit message: {commit_message}")

    issue_numbers = extract_issue_numbers(commit_message)
    print(f"Found issue numbers: {issue_numbers}")

    if issue_numbers:
        close_issues(issue_numbers)
    else:
        print("No issues to close")
