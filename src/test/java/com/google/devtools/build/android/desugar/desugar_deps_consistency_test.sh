#!/bin/bash
#
# Copyright 2016 The Bazel Authors. All rights reserved.
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
set -eu
set -o pipefail

SINGLEJAR="$1"
shift
JAR="$1"
shift

out="$(mktemp)"
if ! "${SINGLEJAR}" --output "${out}" --check_desugar_deps --sources "$@"; then
  # If output was generated, clean it up
  if [ -e "${out}" ]; then rm "${out}"; fi
  case "$0" in
    *_fail_test) echo "Singlejar failed as expected!"; exit 0;;
  esac
  echo "Singlejar unexpectedly failed"
  exit 1
fi

case "$0" in
  *_fail_test) rm "${out}"; echo "Singlejar unexpectedly succeeded :("; exit 1;;
esac

if "${JAR}" tf "${out}" | grep 'desugar_deps'; then
  rm "${out}"
  echo "Singlejar output unexpectedly contains desugaring metadata"
  exit 2
fi  # else grep didn't find anything -> pass
rm "${out}"
exit 0
