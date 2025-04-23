#!/usr/bin/env bash
# -*- coding: utf-8 -*-

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

# Unit tests for pkg_tar

DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source ${DIR}/testenv.sh || { echo "testenv.sh not found!" >&2; exit 1; }

function get_tar_listing() {
  local input=$1
  local test_data="${TEST_DATA_DIR}/${input}"
  # We strip unused prefixes rather than dropping "v" flag for tar, because we
  # want to preserve symlink information.
  tar tvf "${test_data}" | sed -e 's/^.*:00 //'
}

function get_tar_verbose_listing() {
  local input=$1
  local test_data="${TEST_DATA_DIR}/${input}"
  TZ="UTC" tar tvf "${test_data}"
}

function get_tar_owner() {
  local input=$1
  local file=$2
  local test_data="${TEST_DATA_DIR}/${input}"
  tar tvf "${test_data}" | grep "00 $file\$" | cut -d " " -f 2
}

function get_numeric_tar_owner() {
  local input=$1
  local file=$2
  local test_data="${TEST_DATA_DIR}/${input}"
  tar --numeric-owner -tvf "${test_data}" | grep "00 $file\$" | cut -d " " -f 2
}

function get_tar_permission() {
  local input=$1
  local file=$2
  local test_data="${TEST_DATA_DIR}/${input}"
  tar tvf "${test_data}" | fgrep "00 $file" | cut -d " " -f 1
}

function assert_content() {
  local listing="./
./etc/
./etc/nsswitch.conf
./usr/
./usr/titi"
  check_eq "$listing" "$(get_tar_listing $1)"
  check_eq "-rwxr-xr-x" "$(get_tar_permission $1 ./usr/titi)"
  check_eq "-rwxr-xr-x" "$(get_tar_permission $1 ./etc/nsswitch.conf)"
}

function test_tar() {
  local listing="./
./etc/
./etc/nsswitch.conf
./usr/
./usr/titi"
  assert_content "test-tar.tar"
}


run_suite "build_test"
