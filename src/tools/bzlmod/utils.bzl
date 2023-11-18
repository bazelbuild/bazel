# Copyright 2023 The Bazel Authors. All rights reserved.
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

"""Helper functions for Bzlmod build"""

def get_canonical_repo_name(apparent_repo_name):
    """Returns the canonical repo name for the given apparent repo name seen by the module this bzl file belongs to."""
    if not apparent_repo_name.startswith("@"):
        apparent_repo_name = "@" + apparent_repo_name
    return Label(apparent_repo_name).workspace_name

def extract_url(attributes):
    """Extracts the url from the given attributes.

    Args:
        attributes: The attributes to extract the url from.

    Returns:
        The url extracted from the given attributes.
    """
    if "urls" in attributes:
        return attributes["urls"][0]
    elif "url" in attributes:
        return attributes["url"]
    else:
        fail("Could not find url in attributes %s" % attributes)

def parse_http_artifacts(ctx, lockfile_path, required_repos):
    """Parses the http artifacts required from for fetching the given repos from the lockfile.

    Args:
        ctx: the repository / module extension ctx object.
        lockfile_path: The path of the lockfile to extract the http artifacts from.
        required_repos: The list of required repos to extract the http artifacts for,
                        only support `http_archive`, `http_file` and `http_jar` repo rules.

    Returns:
        A list of http artifacts in the form of
        [{"integrity": <integrity value>, "url": <url>}, {"sha256": <sha256 value>, "url": <url>}, ...]
    """
    lockfile = json.decode(ctx.read(lockfile_path))
    http_artifacts = []
    found_repos = []
    for _, module in lockfile["moduleDepGraph"].items():
        if "repoSpec" in module and module["repoSpec"]["ruleClassName"] == "http_archive":
            repo_spec = module["repoSpec"]
            attributes = repo_spec["attributes"]
            repo_name = attributes["name"]

            if repo_name not in required_repos:
                continue
            found_repos.append(repo_name)

            http_artifacts.append({
                "integrity": attributes["integrity"],
                "url": extract_url(attributes),
            })
            if "remote_patches" in attributes:
                for patch, integrity in attributes["remote_patches"].items():
                    http_artifacts.append({
                        "integrity": integrity,
                        "url": patch,
                    })

    for _, extension_entry in lockfile["moduleExtensions"].items():
        extensions = []
        for _, extension_per_platform in extension_entry.items():
            extensions.append(extension_per_platform)
        for extension in extensions:
            for _, repo_spec in extension["generatedRepoSpecs"].items():
                rule_class = repo_spec["ruleClassName"]
                if rule_class == "http_archive" or rule_class == "http_file" or rule_class == "http_jar":
                    attributes = repo_spec["attributes"]
                    repo_name = attributes["name"]

                    if repo_name not in required_repos:
                        continue
                    found_repos.append(repo_name)

                    http_artifacts.append({
                        "sha256": attributes["sha256"],
                        "url": extract_url(attributes),
                    })

    missing_repos = [repo for repo in required_repos if repo not in found_repos]
    if missing_repos:
        fail("Could not find all required repos, missing: %s" % missing_repos)

    return http_artifacts

def parse_bazel_module_repos(ctx, lockfile_path):
    """Parse repo names of http_archive backed Bazel modules from the given lockfile.

    Args:
        ctx: the repository / module extension ctx object.
        lockfile_path: The path of the lockfile to extract the repo names from.

    Returns:
        A list of canonical repository names
    """

    lockfile = json.decode(ctx.read(lockfile_path))
    repos = []
    for _, module in lockfile["moduleDepGraph"].items():
        if "repoSpec" in module and module["repoSpec"]["ruleClassName"] == "http_archive":
            repo_spec = module["repoSpec"]
            attributes = repo_spec["attributes"]
            repo_name = attributes["name"]
            repos.append(repo_name)
    return repos
