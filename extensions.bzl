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

load("//:repositories.bzl", "embedded_jdk_repositories")
load("//:distdir.bzl", "dist_http_archive", "dist_http_jar")
load("//tools/distributions/debian:deps.bzl", "debian_deps")
load("//src/test/shell/bazel:list_source_repository.bzl", "list_source_repository")
load("//src/main/res:winsdk_configure.bzl", "winsdk_configure")

### Extra dependencies for building Bazel

def _bazel_internal_deps(_ctx):
    embedded_jdk_repositories()
    debian_deps()

bazel_internal_deps = module_extension(implementation = _bazel_internal_deps)

### Extra dependencies for testing Bazel

def _bazel_dev_deps(_ctx):
    list_source_repository(name = "local_bazel_source_list")
    dist_http_archive(name = "bazelci_rules")
    winsdk_configure(name = "local_config_winsdk")

bazel_dev_deps = module_extension(implementation = _bazel_dev_deps)

### Extra dependencies for Bazel Android tools

def _bazel_android_deps(_ctx):
    dist_http_jar(name = "android_gmaven_r8")
    dist_http_archive(name = "desugar_jdk_libs")

bazel_android_deps = module_extension(implementation = _bazel_android_deps)
