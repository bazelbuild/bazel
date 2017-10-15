# Copyright 2015 The Bazel Authors. All rights reserved.
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
"""Rules for manipulation Docker images."""

load(":build.bzl", "docker_build")
load(":bundle.bzl", "docker_bundle")

print("The docker_{build,bundle} rules bundled with Bazel are deprecated " +
      "in favor of:\nhttps://github.com/bazelbuild/rules_docker. " +
      "Please change BUILD loads to reference: " +
      "@io_bazel_rules_docker//docker:docker.bzl and add the following to " +
      "your WORKSPACE:\n" +
      """git_repository(
    name = "io_bazel_rules_docker",
    remote = "https://github.com/bazelbuild/rules_docker.git",
    commit = "...",
)
load("@io_bazel_rules_docker//docker:docker.bzl", "docker_repositories")
docker_repositories()""")
