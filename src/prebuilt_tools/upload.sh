#!/usr/bin/env bash

# Copyright 2026 The Bazel Authors. All rights reserved.
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

# Builds the prebuilt tools archive for the current platform, or the platform
# selected by the given Bazel flags, and uploads it to
# gs://bazel-mirror/bazel_prebuilt_tools/tmp/<commit hash>/.
#
# Used by the prebuilt_tools Buildkite pipeline, for example:
# $ src/prebuilt_tools/upload.sh
# $ src/prebuilt_tools/upload.sh --config=windows_arm64

set -euo pipefail

case "$(uname -s | tr '[:upper:]' '[:lower:]')" in
msys*|mingw*|cygwin*)
  declare -r is_windows=true
  export MSYS_NO_PATHCONV=1
  export MSYS2_ARG_CONV_EXCL="*"
  ;;
*)
  declare -r is_windows=false
  ;;
esac

commit_hash="$(git rev-parse HEAD)"
bazel_version="$(bazel info release | cut -d' ' -f2)"

bazel build -c opt "$@" \
  //src/prebuilt_tools:platform \
  //src/prebuilt_tools:prebuilt_tools_zip
bazel_bin="$(bazel info -c opt "$@" bazel-bin)"
platform="$(tr -d '\r\n ' <"${bazel_bin}/src/prebuilt_tools/platform.txt")"

tmp_dir="$(mktemp -d -t 'tmp_bazel_prebuilt_tools_XXXXXX')"
trap 'rm -fr "$tmp_dir"' EXIT
zip_file="${tmp_dir}/prebuilt_tools_${platform}.zip"
cp "${bazel_bin}/src/prebuilt_tools/prebuilt_tools.zip" "$zip_file"
chmod +w "$zip_file"

cat >"${tmp_dir}/README.md" <<EOT
These tools were built from the Bazel repository at commit ${commit_hash}
using Bazel ${bazel_version}. To build the same archive from source, run:

$ git clone https://github.com/bazelbuild/bazel.git
$ cd bazel
$ git checkout ${commit_hash}
$ bazel build -c opt $* //src/prebuilt_tools:prebuilt_tools_zip
EOT
(cd "$tmp_dir" && zip -q "$zip_file" README.md)

gcloud_cmd="gcloud"
if "$is_windows"; then
  gcloud_cmd="gcloud.cmd"
  zip_file="$(cygpath -m "$zip_file")"
fi
"$gcloud_cmd" storage cp "$zip_file" \
  "gs://bazel-mirror/bazel_prebuilt_tools/tmp/${commit_hash}/prebuilt_tools_${platform}.zip"
