#!/bin/bash

# Copyright 2019 The Bazel Authors. All rights reserved.
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

# A script to upload a given java_tools zip on GCS. Used by the java_tools_binaries
# Buildkite pipeline. It is not recommended to run this script manually.
#
# Mandatory flags:
# --java_tools_zip       The workspace-relative path of a java_tools zip.
# --gcs_java_tools_dir   The directory under bazel_java_tools on GCS where the zip is uploaded.
# --java_version         The version of the javac the given zip embeds.
# --platform             The name of the platform where the zip was built.

# Script used by the "java_tools binaries" Buildkite pipeline to build the java tools archives
# and upload them on GCS.
#
# The script has to be executed directly without invoking bazel:
# $ src/upload_all_java_tools.sh
#
# The script cannot be invoked through a sh_binary using bazel because git
# cannot be used through a sh_binary.

commit_hash=$(git rev-parse HEAD)
timestamp=$(date +%s)
bazel_version=$(bazel version | grep 'Build label' | cut -d' ' -f 3)

# Passing the same commit_hash and timestamp to all targets to mark all the artifacts
# uploaded on GCS with the same identifier.
bazel run src:upload_java_tools_java10 -- --commit_hash ${commit_hash} --timestamp ${timestamp} --bazel_version ${bazel_version}
bazel run src:upload_java_tools_java9 -- --commit_hash ${commit_hash} --timestamp ${timestamp} --bazel_version ${bazel_version}
bazel run src:upload_java_tools_dist_java10 -- --commit_hash ${commit_hash} --timestamp ${timestamp} --bazel_version ${bazel_version}
bazel run src:upload_java_tools_dist_java9 -- --commit_hash ${commit_hash} --timestamp ${timestamp} --bazel_version ${bazel_version}
