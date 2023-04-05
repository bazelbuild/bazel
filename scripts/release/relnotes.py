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

import requests


def get_last_release():
  """Discovers the last stable release name from GitHub."""
  response = requests.get("https://github.com/bazelbuild/bazel/releases/latest")
  return response.url.split("/")[-1]


def git(*args):
  """Runs git as a subprocess, and returns its stdout as a list of lines."""
  return subprocess.check_output(["git"] +
                                 list(args)).decode("utf-8").strip().split("\n")


def extract_relnotes(commit_message_lines):
  """Extracts relnotes from a commit message (passed in as a list of lines)."""
  relnote_lines = []
  in_relnote = False
  for line in commit_message_lines:
    if not line or line.startswith("PiperOrigin-RevId:"):
      in_relnote = False
    m = re.match(r"^RELNOTES(?:\[(INC|NEW)\])?:", line)
    if m is not None:
      in_relnote = True
      line = line[len(m[0]):]
      if m[1] == "INC":
        line = "**[Incompatible]** " + line.strip()
    line = line.strip()
    if in_relnote and line:
      relnote_lines.append(line)
  relnote = " ".join(relnote_lines)
  relnote_lower = relnote.strip().lower().rstrip(".")
  if relnote_lower == "n/a" or relnote_lower == "none":
    return None
  return relnote


def get_relnotes_between(base, head):
  """Gets all relnotes for commits between `base` and `head`."""
  commits = git("rev-list", f"{base}..{head}", "--grep=RELNOTES")
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
    relnote = extract_relnotes(lines)
    if relnote is not None:
      relnotes.append(relnote)
  return relnotes


def get_external_authors_between(base, head):
  """Gets all external authors for commits between `base` and `head`."""
  authors = git("log", f"{base}..{head}", "--format=%aN|%aE")
  authors = set(author.partition("|")[0].rstrip() for author in authors
                if not author.endswith("@google.com"))
  return ", ".join(sorted(authors, key=str.casefold))


if __name__ == "__main__":
  # Get the last stable release.
  last_release = get_last_release()
  print("last_release is", last_release)
  git("fetch", "origin", f"refs/tags/{last_release}:refs/tags/{last_release}")

  # Assuming HEAD is on the current (to-be-released) release, find the merge
  # base with the last release so that we know which commits to generate notes
  # for.
  merge_base = git("merge-base", "HEAD", last_release)[0]
  print("merge base with", last_release, "is", merge_base)

  # Generate notes for all commits from last branch cut to HEAD, but filter out
  # any identical notes from the previous release branch.
  cur_release_relnotes = get_relnotes_between(merge_base, "HEAD")
  last_release_relnotes = set(get_relnotes_between(merge_base, last_release))
  filtered_relnotes = [
      note for note in cur_release_relnotes if note not in last_release_relnotes
  ]

  # Reverse so that the notes are in chronological order.
  filtered_relnotes.reverse()
  print()
  print()
  for note in filtered_relnotes:
    print("*", note)

  print()
  print()
  external_authors = get_external_authors_between(merge_base, "HEAD")
  print(
      "This release contains contributions from many people at Google, "
      f"as well as {external_authors}.")
