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

set -euo pipefail

# --- begin runfiles.bash initialization ---
if [[ ! -d "${RUNFILES_DIR:-/dev/null}" && ! -f "${RUNFILES_MANIFEST_FILE:-/dev/null}" ]]; then
    if [[ -f "$0.runfiles_manifest" ]]; then
      export RUNFILES_MANIFEST_FILE="$0.runfiles_manifest"
    elif [[ -f "$0.runfiles/MANIFEST" ]]; then
      export RUNFILES_MANIFEST_FILE="$0.runfiles/MANIFEST"
    elif [[ -f "$0.runfiles/bazel_tools/tools/bash/runfiles/runfiles.bash" ]]; then
      export RUNFILES_DIR="$0.runfiles"
    fi
fi
if [[ -f "${RUNFILES_DIR:-/dev/null}/bazel_tools/tools/bash/runfiles/runfiles.bash" ]]; then
  source "${RUNFILES_DIR}/bazel_tools/tools/bash/runfiles/runfiles.bash"
elif [[ -f "${RUNFILES_MANIFEST_FILE:-/dev/null}" ]]; then
  source "$(grep -m1 "^bazel_tools/tools/bash/runfiles/runfiles.bash " \
            "$RUNFILES_MANIFEST_FILE" | cut -d ' ' -f 2-)"
else
  echo >&2 "ERROR: cannot find @bazel_tools//tools/bash/runfiles:runfiles.bash"
  exit 1
fi
# --- end runfiles.bash initialization ---

# Parsing the flags.
while [[ -n "$@" ]]; do
  arg="$1"; shift
  val="$1"; shift
  case "$arg" in
    "--java_tools_zip") java_tools_zip_name="$val" ;;
    "--gcs_java_tools_dir") gcs_java_tools_dir="$val" ;;
    "--java_version") java_version="$val" ;;
    "--platform") platform="$val" ;;
    "--commit_hash") commit_hash="$val" ;;
    "--timestamp") timestamp="$val" ;;
    "--bazel_version") bazel_version="$val" ;;
    *) echo "Flag $arg is not recognized." && exit 1 ;;
  esac
done

java_tools_zip=$(rlocation io_bazel/${java_tools_zip_name})

tmp_dir=$(mktemp -d -t 'tmp_bazel_zip_files_XXXXX')
trap "rm -fr $tmp_dir" EXIT
tmp_zip="$tmp_dir/archive.zip"

cp $java_tools_zip $tmp_zip
chmod +w $tmp_zip

# Create the README.md file.
readme_file="README.md"
cat >${readme_file} <<EOF
This Java tools version was built from the bazel repository at commit hash ${commit_hash}
using bazel version ${bazel_version}.
To build from source the same zip run the commands:

$ git clone https://github.com/bazelbuild/bazel.git
$ git checkout ${commit_hash}
$ bazel build //src:java_tools_java${java_version}.zip
EOF

zip -rv "${tmp_zip}" "${readme_file}"

gsutil_cmd="gsutil"
if [[ "$platform" == "windows" ]]; then
  gsutil_cmd="gsutil.cmd"
fi

"$gsutil_cmd" cp "$tmp_zip" \
 "gs://bazel-mirror/bazel_java_tools/${gcs_java_tools_dir}/java_tools_javac${java_version}_${platform}-${commit_hash}-${timestamp}.zip"
