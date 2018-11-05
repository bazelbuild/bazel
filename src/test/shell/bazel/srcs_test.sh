#!/bin/bash
#
# Copyright 2017 The Bazel Authors. All rights reserved.
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
#
# Test that all sources in Bazel are contained in the //:srcs filegroup
# Actually this test just compares the two input (the file list and the
# //:srcs filegroup and show the diff)

LIST_SRCS="${TEST_SRCDIR}/local_bazel_source_list/sources.txt"
SRCS_QUERY="$(mktemp)"

# Rewrite labels to file paths. This assumes any external repo is actually
# a local_repository located in third_party.
cat "${TEST_SRCDIR}/io_bazel/src/test/shell/bazel/srcs_list" \
  | sed -e 's|@\([^/]*\)//|third_party/\1|' \
  | sed -e 's|^//||' | sed -e 's|^:||' | sed -e 's|:|/|' \
  | sort -u >"${SRCS_QUERY}"

res="$(diff -U 0 "${LIST_SRCS}" "${SRCS_QUERY}" | sed -e 's|^-||' \
  | grep -Ev '^(@@|\+\+|--)' || true)"

if [ -n "${res}" ]; then
  echo "//:srcs filegroup do not contains all the sources, missing:
${res}"
  exit 1
fi
