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

load("@buildozer//:buildozer.bzl", "BUILDOZER_LABEL")

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
    if "moduleDepGraph" in lockfile:
        # TODO: Remove this branch after Bazel is built with 7.2.0.
        for _, module in lockfile["moduleDepGraph"].items():
            if "repoSpec" in module and module["repoSpec"]["ruleClassName"] == "http_archive":
                repo_spec = module["repoSpec"]
                attributes = repo_spec["attributes"]
                repo_name = _module_repo_name(module)

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
    else:
        for url, sha256 in lockfile["registryFileHashes"].items():
            if not url.endswith("/source.json"):
                continue
            segments = url.split("/")
            module = {
                "name": segments[-3],
                "version": segments[-2],
            }
            repo_name = _module_repo_name(module)
            if repo_name not in required_repos:
                continue
            found_repos.append(repo_name)

            ctx.delete("./tempfile")
            ctx.download(url, "./tempfile", executable = False, sha256 = sha256)
            source_json = json.decode(ctx.read("./tempfile"))

            http_artifacts.append({
                "integrity": source_json["integrity"],
                "url": source_json["url"],
            })

            for patch, integrity in source_json.get("patches", {}).items():
                http_artifacts.append({
                    "integrity": integrity,
                    "url": url.rsplit("/", 1)[0] + "/patches/" + patch,
                })

    for extension_id, extension_entry in lockfile["moduleExtensions"].items():
        if extension_id.startswith("@@"):
            # @@rules_foo~//:extensions.bzl%foo --> rules_foo~
            module_repo_name = extension_id.removeprefix("@@").partition("//")[0]
        else:
            # //:extensions.bzl%foo --> _main
            module_repo_name = "_main"
        extension_name = extension_id.partition("%")[2]
        repo_name_prefix = "{}~{}~".format(module_repo_name, extension_name)
        extensions = []
        for _, extension_per_platform in extension_entry.items():
            extensions.append(extension_per_platform)
        for extension in extensions:
            for local_name, repo_spec in extension["generatedRepoSpecs"].items():
                rule_class = repo_spec["ruleClassName"]

                # TODO(pcloudy): Remove "kotlin_compiler_repository" after https://github.com/bazelbuild/rules_kotlin/issues/1106 is fixed
                if rule_class == "http_archive" or rule_class == "http_file" or rule_class == "http_jar" or rule_class == "kotlin_compiler_repository":
                    attributes = repo_spec["attributes"]
                    repo_name = repo_name_prefix + local_name

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

BCR_URL_SCHEME = "https://bcr.bazel.build/modules/{name}/{version}/{file}"

def parse_registry_files(ctx, lockfile_path, module_files):
    """Parses the registry files referenced by the given lockfile and returns them in http_file form.

    Args:
        ctx: the repository / module extension ctx object.
        lockfile_path: The path of the lockfile to extract the registry files from.
        module_files: The paths of non-registry module files to use during fake module resolution.

    Returns:
        A list of http artifacts in the form of
        [{"sha256": <sha256 value>, "url": <url>}, ...]
    """
    lockfile = json.decode(ctx.read(lockfile_path))
    registry_file_hashes = lockfile.get("registryFileHashes", {})
    if registry_file_hashes:
        return [
            {"sha256": sha256, "url": url}
            for url, sha256 in registry_file_hashes.items()
        ]

    # TODO: Remove the following code after Bazel is built with 7.2.0.
    registry_files = ["https://bcr.bazel.build/bazel_registry.json"]

    # 1. Collect all source.json files of selected module versions.
    for module in lockfile["moduleDepGraph"].values():
        if module["version"]:
            registry_files.append(BCR_URL_SCHEME.format(
                name = module["name"],
                version = module["version"],
                file = "source.json",
            ))

    # 2. Download registry files to compute their hashes.
    registry_file_artifacts = []
    downloads = {
        url: ctx.download(url, "./tempdir/{}".format(i), executable = False, block = False)
        for i, url in enumerate(registry_files)
    }
    for url, download in downloads.items():
        hash = download.wait()
        registry_file_artifacts.append({"url": url, "sha256": hash.sha256})

    # 3. Perform module resolution in Starlark to get the MODULE.bazel file URLs
    #    of all module versions relevant during resolution. The lockfile only
    #    contains the selected module versions.
    module_file_stack = [ctx.path(module_file) for module_file in module_files]
    seen_deps = {}
    for _ in range(1000000):
        if not module_file_stack:
            break
        bazel_deps = _extract_bazel_deps(ctx, module_file_stack.pop())
        downloads = {}
        for dep in bazel_deps:
            if dep in seen_deps:
                continue
            url = BCR_URL_SCHEME.format(
                name = dep.name,
                version = dep.version,
                file = "MODULE.bazel",
            )
            path = ctx.path("./tempdir/modules/{name}/{version}/MODULE.bazel".format(
                name = dep.name,
                version = dep.version,
            ))
            module_file_stack.append(path)
            seen_deps[dep] = None
            downloads[url] = ctx.download(url, path, executable = False, block = False)

        for url, download in downloads.items():
            hash = download.wait()
            registry_file_artifacts.append({"url": url, "sha256": hash.sha256})

    ctx.delete("./tempdir")
    return registry_file_artifacts

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
    for url in lockfile["registryFileHashes"].keys():
        if not url.endswith("/source.json"):
            continue
        segments = url.split("/")
        module = {
            "name": segments[-3],
            "version": segments[-2],
        }
        repo_name = _module_repo_name(module)
        repos.append(repo_name)
    return {repo: None for repo in repos}.keys()

# Keep in sync with ModuleKey.
_WELL_KNOWN_MODULES = ["bazel_tools", "local_config_platform", "platforms"]

def _module_repo_name(module):
    module_name = module["name"]
    if module_name in _WELL_KNOWN_MODULES:
        return module_name

    # TODO(pcloudy): Simplify the following logic after we upgrade to 7.1
    if get_canonical_repo_name("rules_cc").endswith("~"):
        return "{}~".format(module_name)

    return "{}~{}".format(module_name, module["version"])

def _extract_bazel_deps(ctx, module_file):
    buildozer = ctx.path(BUILDOZER_LABEL)
    temp_path = "tempdir/buildozer/MODULE.bazel"
    ctx.delete(temp_path)
    ctx.symlink(module_file, temp_path)
    result = ctx.execute([buildozer, "print name version dev_dependency", temp_path + ":%bazel_dep"])
    if result.return_code != 0:
        fail("Failed to extract bazel_dep from {}:\n{}".format(module_file, result.stderr))
    deps = []
    for line in result.stdout.splitlines():
        if "  " in line:
            # The dep doesn't have a version specified, which is only valid in
            # the root module. Ignore it.
            continue
        if line.endswith(" True"):
            # The dep is a dev_dependency, ignore it.
            continue
        name, version, _ = line.split(" ")
        deps.append(struct(name = name, version = version))

    return deps
