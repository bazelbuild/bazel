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

"""Module extensions for loading dependencies we need to build Bazel.

"""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_file")
load("//:distdir.bzl", "dist_http_archive", "repo_cache_tar")
load("//:distdir_deps.bzl", "DIST_ARCHIVE_REPOS", "TEST_REPOS")
load("//:repositories.bzl", "embedded_jdk_repositories")
load("//src/main/res:winsdk_configure.bzl", "winsdk_configure")
load("//src/test/shell/bazel:list_source_repository.bzl", "list_source_repository")
load("//src/tools/bzlmod:utils.bzl", "parse_http_artifacts")
load("//tools/distributions/debian:deps.bzl", "debian_deps")

### Dependencies for building Bazel
def _bazel_build_deps(_ctx):
    embedded_jdk_repositories()
    debian_deps()
    repo_cache_tar(name = "bootstrap_repo_cache", repos = DIST_ARCHIVE_REPOS, dirname = "derived/repository_cache")

bazel_build_deps = module_extension(implementation = _bazel_build_deps)

### Dependencies for testing Bazel
def _bazel_test_deps(_ctx):
    list_source_repository(name = "local_bazel_source_list")
    dist_http_archive(name = "bazelci_rules")
    winsdk_configure(name = "local_config_winsdk")

bazel_test_deps = module_extension(implementation = _bazel_test_deps)

_HUB_TEST_REPO_BUILD = """
filegroup(
  name="srcs",
  srcs = {srcs},
  visibility = ["//visibility:public"],
)
"""

def _hub_test_repo_impl(ctx):
    ctx.file(
        "BUILD",
        _HUB_TEST_REPO_BUILD.format(srcs = ["@%s//file" % repo for repo in ctx.attr.repos]),
    )

hub_test_repo = repository_rule(
    implementation = _hub_test_repo_impl,
    attrs = {"repos": attr.string_list()},
)

def _test_repo_extension_impl(ctx):
    """This module extension is used to fetch http artifacts required for generating TEST_REPOS."""
    lockfile_path = ctx.path(Label("//:MODULE.bazel.lock"))
    http_artifacts = parse_http_artifacts(ctx, lockfile_path, TEST_REPOS)
    name = "test_repo_"
    cnt = 1
    for artifact in http_artifacts:
        # Define one http_file for each artifact so that we can fetch them in parallel.
        http_file(
            name = name + str(cnt),
            url = artifact["url"],
            sha256 = artifact["sha256"] if "sha256" in artifact else None,
            integrity = artifact["integrity"] if "integrity" in artifact else None,
        )
        cnt += 1

    # write a repo rule that depends on all the http_file rules
    hub_test_repo(name = "test_repos", repos = [(name + str(i)) for i in range(1, cnt)])

test_repo_extension = module_extension(implementation = _test_repo_extension_impl)

### Dependencies for Bazel Android tools
def _bazel_android_deps(_ctx):
    dist_http_archive(name = "desugar_jdk_libs")

bazel_android_deps = module_extension(implementation = _bazel_android_deps)
