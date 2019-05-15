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

# A script that creates java_tools release candidates or release artifacts.
#
# Before creating a release candidate the script assumes that the java_tools
# binaries pipeline was previously run and generated java_tools artifacts at
# a commit hash.
#
# The script is using gsutil to copy artifacts.
#
# Mandatory flags:

# --java_version        The JDK version included in the java_tools to be released.
# --java_tools_version  The version number of the java_tools to be released.
# --rc                  The release candidate number of current release.
#                       If --release true then --rc is the number of the rc to
#                       be released.
#                       If --release false then --rc is the number of the rc to
#                       be created.
# --release             "true" if the script has to create the release artifact
#                       or "false" if the script has to create a release
#                       candidate.
# --commit_hash         The commit hash where the java_tools binaries pipeline
#                       was run. Mandatory only if --release false.
#
# Usage examples:
#
# To create the first release candidate for a new java_tools version 2.1 and
# JDK11, that was built at commit_hash 123456:
#
# src/create_java_tools_release.sh --commit_hash 123456 \
#     --java_tools_version 2.1 --java_version 11 --rc 1 --release false
#
# To release the release candidate created above:
#
# src/create_java_tools_release.sh \
#     --java_tools_version 2.1 --java_version 11 --rc 1 --release true

set -euo pipefail

# Parsing the flags.
while [[ -n "$@" ]]; do
  arg="$1"; shift
  val="$1"; shift
  case "$arg" in
    "--java_version") java_version="$val" ;;
    "--java_tools_version") java_tools_version="$val" ;;
    "--commit_hash") commit_hash="$val" ;;
    "--rc") rc="$val" ;;
    "--release") release="$val" ;;
    *) echo "Flag $arg is not recognized." && exit 1 ;;
  esac
done

tmp_dir=$(mktemp -d -t 'tmp_bazel_zip_files_XXXXX')
trap "rm -fr $tmp_dir" EXIT

for platform in linux windows darwin; do
  rc_url="gs://bazel-mirror/bazel_java_tools/release_candidates/javac${java_version}/v${java_tools_version}/java_tools_javac${java_version}_${platform}-v${java_tools_version}-rc${rc}.zip"

  if [[ $release == "true" ]]; then
    release_url="gs://bazel-mirror/bazel_java_tools/releases/javac${java_version}/v${java_tools_version}/java_tools_javac${java_version}_${platform}-v${java_tools_version}.zip"
    # Make release candidate the release artifact for the current platform.
    gsutil -q cp ${rc_url} ${release_url}
    release_artifact="releases/javac${java_version}/v${java_tools_version}/java_tools_javac${java_version}_${platform}-v${java_tools_version}.zip"
  else
    tmp_url=$(gsutil ls -lh gs://bazel-mirror/bazel_java_tools/tmp/build/${commit_hash}/java${java_version}/java_tools_javac${java_version}_${platform}* | sort -k 2 | grep "gs" | cut -d " " -f 7)
    # Make the generated artifact a release candidate for the current platform.
    gsutil -q cp ${tmp_url} ${rc_url}
    release_artifact="release_candidates/javac${java_version}/v${java_tools_version}/java_tools_javac${java_version}_${platform}-v${java_tools_version}-rc${rc}.zip"
  fi

  local_zip="$tmp_dir/java_tools_$platform.zip"
  # Download the file to compute its sha256sum.
  gsutil -q cp ${rc_url} ${local_zip}
  file_hash=$(sha256sum ${local_zip} | cut -d' ' -f1)
  echo "${release_artifact} ${file_hash}"
done
