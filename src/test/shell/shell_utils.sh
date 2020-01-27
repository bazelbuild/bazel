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

set -euo pipefail

PLATFORM="$(uname -s | tr 'A-Z' 'a-z')"

# Print the resolved path of a symbolic link (or the path itself if not a link).
#
# The result may be relative to the current working directory and may contain
# "." and ".." references. Use `normalize_path` to remove those.
#
# Fail and print nothing if the input path doesn't exist, i.e. it's a
# non-existent file, dangling symlink, or circular symlink.
function resolve_links() {
  local name="$1"

  if [ -e "$name" ]; then
    # resolve all links, keep path absolute
    while [ -L "$name" ]; do
      local target=$(readlink "$name")
      if [ "$(echo "$target" | head -c1)" = "/" ]; then
        name="$target"
      else
        name="$(dirname "$name")/$target"
      fi
    done
    echo "$name"
  else
    false  # fail the function
  fi
}

# Normalize a path string by removing "." and ".." references from it.
#
# Print the result to stdout.
#
# The path doesn't have to point to an existing file.
function normalize_path() {
  local name="$1"

  local path=""
  local uplevels=0
  while [ "$name" != "/" -a "$name" != "." ]; do
    local segment="$(basename "$name")"
    name="$(dirname "$name")"

    [ "$segment" = "." ] && continue

    if [ "$segment" = ".." ]; then
      uplevels="$((${uplevels}+1))"
    else
      if [ "$uplevels" -gt 0 ]; then
        uplevels="$((${uplevels}-1))"
      else
        path="${segment}/${path}"
      fi
    fi
  done

  if [ "$name" = "." ]; then
    while [ "$uplevels" -gt 0 ]; do
      path="../$path"
      uplevels="$((${uplevels}-1))"
    done
  fi

  path="${path%/}"

  if [ "$name" = "/" ]; then
    echo "/$path"
  else
    [ -n "$path" ] && echo "$path" || echo "."
  fi
}

# Custom implementation of `realpath(1)`.
#
# Resolves a symlink to the absolute path it points to. Fails if the link or
# its target does not exist.
#
# Fails and prints nothing if the input path doesn't exist, i.e. it's a
# non-existent file, dangling symlink, or circular symlink.
#
# We use this method instead of `realpath(1)` because the latter is not
# available on Mac OS X by default.
function get_real_path() {
  local name="$1"
  name="$(resolve_links "$name" || echo)"
  if [ -n "$name" ]; then
    normalize_path "$(pwd)/$name"
  else
    false  # fail the function
  fi
}

function md5_file() {
  if [ $# -gt 0 ]; then
    local result=""
    if [[ ${PLATFORM} == "darwin" ]] || [[ ${PLATFORM} == "freebsd" ]] || [[ ${PLATFORM} == "openbsd" ]]; then
      result=$(md5 -q $@ || echo)
    else
      result=$(md5sum $@ | awk '{print $1}' || echo)
    fi

    if [ -n "$result" ]; then
      echo "$result"
    else
      false
    fi
  else
    false
  fi
}

# Usage: expect_query_targets <arguments>
# Checks that log file contains exactly the targets in the argument list.
function expect_query_targets() {
  for arg in $@; do
    expect_log_once "^$arg$"
  done

# Checks that the number of lines started with '//' or '@' equals to the number of
# arguments provided.
  expect_log_n "^(//|@)\S*$" $#
}

# Usage: expect_log_once <regexp> [error-message]
# Asserts that $TEST_log contains one line matching <regexp>.
# Prints the contents of $TEST_log and the specified (optional)
# error message otherwise, and returns non-zero.
function expect_log_once() {
    local pattern=$1
    local message=${2:-Expected regexp "$pattern" not found exactly once}
    expect_log_n "$pattern" 1 "$message"
}

# Usage: expect_log_n <regexp> <count> [error-message]
# Asserts that $TEST_log contains <count> lines matching <regexp>.
# Prints the contents of $TEST_log and the specified (optional)
# error message otherwise, and returns non-zero.
function expect_log_n() {
    local pattern=$1
    local expectednum=${2:-1}
    local message=${3:-Expected regexp "$pattern" not found exactly $expectednum times}
    local count=$(grep -sc -E "$pattern" $TEST_log)
    [[ $count = $expectednum ]] && return 0
    fail "$message"
    return 1
}
