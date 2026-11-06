# Copyright 2022 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script to generate release notes."""

import re
import subprocess
import sys
import requests


def git(*args):
  """Runs git as a subprocess, and returns its stdout as a list of lines."""
  return subprocess.check_output(["git"] +
                                 list(args)).decode("utf-8").strip().split("\n")


def extract_title(commit_message_lines):
  """Extracts first line from commit message (passed in as a list of lines)."""
  return re.sub(
      r"\[\d+\.\d+\.\d\]\s?", "", commit_message_lines[0].strip()
  )


def extract_relnotes(commit_message_lines):
  """Extracts relnotes from a commit message (passed in as a list of lines)."""
  relnote_lines = []
  in_relnote = False

  title = extract_title(commit_message_lines)
  issue_id = re.search(r"\(\#[0-9]+\)$", title.strip().split()[-1])

  for line in commit_message_lines:
    line = line.strip()
    if (
        not line
        or line.startswith("PiperOrigin-RevId:")
        or re.match(r"^\s*(Fixes|Closes)\s+#\d+\.?\s*$", line)
    ):
      in_relnote = False
    m = re.match(r"^RELNOTES(?:\[(INC|NEW|inc|new)\])?:", line)
    if m is not None:
      in_relnote = True
      line = line[len(m[0]) :]
      if line.strip().lower().rstrip(".") in ["n/a", "na", "none"]:
        return None
      if m[1] == "INC" or m[1] == "inc":
        line = "**[Incompatible]** " + line.strip()
    line = line.strip()
    if in_relnote and line:
      relnote_lines.append(line)
  relnote = " ".join(relnote_lines)

  if issue_id and relnote:
    relnote += " " + issue_id.group(0).strip()

  return relnote


def get_relnotes_between(base, head, is_patch_release):
  """Gets all relnotes for commits between `base` and `head`."""
  commits = git("rev-list", f"{base}..{head}")
  if commits == [""]:
    return []
  relnotes = []
  rolled_back_commits = set()
  # We go in reverse-chronological order, so that we can identify rollback
  # commits and ignore the rolled-back commits.
  for commit in commits:
    if commit in rolled_back_commits:
      continue
    lines = git("show", "-s", commit, "--pretty=format:%B")
    m = re.match(r"^Automated rollback of commit ([\dA-Fa-f]+)", lines[0])
    if m is not None:
      rolled_back_commits.add(m[1])
      # The rollback commit itself is also skipped.
      continue
    relnote = (
        extract_title(lines) if is_patch_release else extract_relnotes(lines)
    )
    if relnote is not None:
      relnotes.append(relnote)
  return relnotes


def get_github_token():
  """Fetches the GitHub API token from GCP Secret Manager."""
  return (
      subprocess.check_output([
          "gcloud",
          "secrets",
          "versions",
          "access",
          "latest",
          "--secret=github-trusted-token",
          "--project=bazel-public",
      ])
      .decode("utf-8")
      .strip()
      .split("\n")[0]
  )


def get_label(issue_id, auth=None):
  """Get team-X label added to issue."""
  if auth is None:
    auth = get_github_token()
  headers = {
      "Authorization": "Bearer " + auth,
      "Accept": "application/vnd.github+json",
  }
  url = (
      f"https://api.github.com/repos/bazelbuild/bazel/issues/{issue_id}/labels"
  )
  try:
    response = requests.get(url, headers=headers, timeout=30)
  except requests.RequestException as e:
    print(
        f"Warning: Failed to fetch labels for #{issue_id}: {e}",
        file=sys.stderr,
    )
    return None

  # GitHub's REST API returns a JSON array (`list` of label dicts) on HTTP 200,
  # but returns a JSON error object (`dict`, e.g. `{"message": ...}`) on non-2xx
  # responses. For instance, HTTP 404 is returned for unknown or external-repo
  # PR numbers, and HTTP 422 ("Could not resolve to a node with the global id
  # of 'PR_...'") is returned when a PR author's GitHub account has been
  # deleted or suspended. Iterating directly over that error dict yields its
  # string keys and raises `AttributeError`, so check `response.ok` first and
  # return `None` to fall back to the "General" category.
  if not response.ok:
    print(
        f"Warning: GitHub API returned HTTP {response.status_code} for"
        f" #{issue_id}; falling back to 'General' category.",
        file=sys.stderr,
    )
    return None

  try:
    labels = response.json()
  except ValueError as e:
    print(
        f"Warning: Invalid JSON in labels response for #{issue_id}: {e}",
        file=sys.stderr,
    )
    return None

  if not isinstance(labels, list):
    return None

  for item in labels:
    if isinstance(item, dict):
      value = item.get("name", "")
      if "team-" in value:
        return value.strip()
  return None


def get_categorized_relnotes(filtered_notes, auth=None):
  """Sort release notes by category."""
  categorized_relnotes = {}
  for relnote in filtered_notes:
    parts = relnote.strip().split()
    if not parts:
      continue
    # Safely access the last element now that we know 'parts' is not empty
    issue_id = re.search(r"\(\#[0-9]+\)$", parts[-1])

    category = None
    if issue_id:
      # Fetch the GitHub token once lazily on the first issue lookup rather than
      # spawning a `gcloud` subprocess inside `get_label` for every issue.
      if auth is None:
        auth = get_github_token()
      category = get_label(
          re.sub(r"\(|\#|\)", "", issue_id.group(0).strip()), auth=auth
      )

    if category is None:
      category = "General"
    else:
      category = re.sub("team-", "", category)

    try:
      categorized_relnotes[category].append(relnote)
    except KeyError:
      categorized_relnotes[category] = [relnote]

  return dict(sorted(categorized_relnotes.items()))


def get_external_authors_between(base, head):
  """Gets all external authors for commits between `base` and `head`."""

  # Get all authors
  authors = git("log", f"{base}..{head}", "--format=%aN|%aE")
  authors = set(
      author.partition("|")[0].rstrip()
      for author in authors if not (author.endswith(("@google.com"))))

  # Get all co-authors
  contributors = git(
      "log", f"{base}..{head}", "--format=%(trailers:key=Co-authored-by)"
  )

  coauthors = []
  for coauthor in contributors:
    if coauthor and not re.search("@google.com", coauthor):
      coauthors.append(
          " ".join(re.sub(r"Co-authored-by: |<.*?>", "", coauthor).split())
      )
  return ", ".join(sorted(authors.union(coauthors), key=str.casefold))


def parse_version(tag):
  """Parses a version string into a tuple of ints for semantic comparison."""
  match = re.search(r"^v?(\d+(?:\.\d+)*)", tag)
  if match:
    return tuple(int(x) for x in match.group(1).split("."))
  return ()


def is_rolling_release(release_name):
  """Returns True if `release_name` is a rolling pre-release (or RC of one)."""
  # Matches the `is_rolling_release` pattern in `common.sh`, e.g.
  # "11.0.0-pre.20261022.3", "11.0.0-pre.20261022.3.1", or "...rc1".
  return bool(
      re.match(
          r"^\d+\.\d+\.\d+-pre\.\d{8}\.\d+(?:\.\d+)?(?:rc[\d.]+)?$",
          release_name,
      )
  )


def get_filtered_notes(base, previous_release, is_patch_release):
  # Generate notes for all commits from last branch cut to HEAD, but filter out
  # any identical notes from the previous release branch.
  cur_release_relnotes = get_relnotes_between(
      base, "HEAD", is_patch_release
  )
  last_release_relnotes = set(
      get_relnotes_between(base, previous_release, is_patch_release)
  )
  return [
      note for note in cur_release_relnotes if note not in last_release_relnotes
  ]


if __name__ == "__main__":
  # Get last release and make sure it's consistent with current X.Y.Z release
  # e.g. if current_release is 5.3.3, last_release should be 5.3.2 even if
  # latest release is 6.1.1
  current_release = git("rev-parse", "--abbrev-ref", "HEAD")[0]

  if current_release.startswith("release-"):
    current_release = re.sub(r"rc\d+", "", current_release[len("release-") :])
  else:
    try:
      current_release = git("describe", "--tags")[0]
    except subprocess.CalledProcessError:
      print("Error: Not a release branch.", file=sys.stderr)
      sys.exit(1)

  # Rolling releases (`X.Y.Z-pre.YYYYMMDD.N`) do not announce categorized
  # release notes via `relnotes.py` (consistent with `relnotes.sh`, which only
  # invokes `relnotes.py` when `is_rolling_release` is false). Exiting early
  # also avoids misclassifying a rolling pre-release as a patch release below
  # (`not current_release.endswith(".0")`) and querying GitHub labels for every
  # commit title since the previous full release tag.
  if is_rolling_release(current_release):
    print(
        "Skipping release notes generation for rolling release"
        f" {current_release}."
    )
    sys.exit(0)

  is_patch = not current_release.endswith(".0")

  tags = [
      tag
      for tag in git("tag", "--sort=version:refname")
      if "pre" not in tag and "rc" not in tag
  ]

  # Get the baseline for RCs (before release tag is created)
  if current_release not in tags:
    tags.append(current_release)

  tags.sort(key=parse_version)
  last_release = tags[tags.index(current_release) - 1]

  # Assuming HEAD is on the current (to-be-released) release, find the merge
  # base with the last release so that we know which commits to generate notes
  # for.
  merge_base = git("merge-base", "HEAD", last_release)[0]

  filtered_relnotes = get_filtered_notes(merge_base, last_release, is_patch)

  # Reverse so that the notes are in chronological order.
  filtered_relnotes.reverse()
  print()
  print("Release Notes:")
  print()

  # For minor releases w/o any RELNOTES tags, follow the patch release format to
  # get release notes (PR title instead of RELNOTES tag)
  # Not applicable for major or patch releases
  if all(not note for note in filtered_relnotes) and is_patch == 0:
    is_patch = True
    filtered_relnotes = get_filtered_notes(merge_base, last_release, is_patch)

  categorized_release_notes = get_categorized_relnotes(filtered_relnotes)
  for label in categorized_release_notes:
    print(label + ":")
    for note in categorized_release_notes[label]:
      print("+", note)
    print()

  print()
  print("Acknowledgements:")
  external_authors = get_external_authors_between(merge_base, "HEAD")
  print()
  print("This release contains contributions from many people at Google" +
        ("." if not external_authors else f", as well as {external_authors}."))
