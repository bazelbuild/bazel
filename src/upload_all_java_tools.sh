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

# Script used by the "java_tools binaries" Buildkite pipeline to build the java tools archives
# and upload them on GCS.
#
# The script has to be executed directly without invoking bazel:
# $ src/upload_all_java_tools.sh
#
# The script cannot be invoked through a sh_binary using bazel because git
# cannot be used through a sh_binary.

set -euo pipefail

case "$(uname -s | tr [:upper:] [:lower:])" in
msys*|mingw*|cygwin*)
  declare -r is_windows=true
  ;;
*)
  declare -r is_windows=false
  ;;
esac

if "$is_windows"; then
  export MSYS_NO_PATHCONV=1
  export MSYS2_ARG_CONV_EXCL="*"
fi

commit_hash=$(git rev-parse HEAD)
timestamp=$(date +%s)
bazel_version=$(bazel info release | cut -d' ' -f2)

# Passing the same commit_hash and timestamp to all targets to mark all the artifacts
# uploaded on GCS with the same identifier.
for java_version in 9 10 11 12; do

    bazel build //src:java_tools_java${java_version}_zip
    zip_path=${PWD}/bazel-bin/src/java_tools_java${java_version}.zip

    if "$is_windows"; then
        # Windows needs "file:///c:/foo/bar".
        file_url="file:///$(cygpath -m ${zip_path})"
    else
        # Non-Windows needs "file:///foo/bar".
        file_url="file://${zip_path}"
    fi
    bazel test --verbose_failures --test_output=all --nocache_test_results \
      //src/test/shell/bazel:bazel_java_test_local_java_tools_jdk${java_version} \
      --define=LOCAL_JAVA_TOOLS_ZIP_URL="${file_url}"

    bazel run //src:upload_java_tools_java${java_version} -- \
        --java_tools_zip src/java_tools_java${java_version}.zip \
        --commit_hash ${commit_hash} \
        --timestamp ${timestamp} \
        --bazel_version ${bazel_version}

    bazel run //src:upload_java_tools_dist_java${java_version} -- \
        --commit_hash ${commit_hash} \
        --timestamp ${timestamp} \
        --bazel_version ${bazel_version}
done
