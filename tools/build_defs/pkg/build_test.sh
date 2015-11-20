#!/bin/bash

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

# Unit tests for pkg_deb and pkg_tar

DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source ${DIR}/testenv.sh || { echo "testenv.sh not found!" >&2; exit 1; }

function get_tar_listing() {
  local input=$1
  local test_data="${TEST_DATA_DIR}/${input}"
  tar tvf "${test_data}" | sed -e 's/^.*:00 //'
}

function get_tar_permission() {
  local input=$1
  local file=$2
  local test_data="${TEST_DATA_DIR}/${input}"
  tar tvf "${test_data}" | fgrep "00 $file" | cut -d " " -f 1
}

function get_deb_listing() {
  local input=$1
  local test_data="${TEST_DATA_DIR}/${input}"
  dpkg-deb -c "${test_data}" | sed -e 's/^.*:00 //'
}

function get_deb_description() {
  local input=$1
  local test_data="${TEST_DATA_DIR}/${input}"
  dpkg-deb -I "${test_data}"
}

function get_deb_permission() {
  local input=$1
  local file=$2
  local test_data="${TEST_DATA_DIR}/${input}"
  dpkg-deb -c "${test_data}" | fgrep "00 $file" | cut -d " " -f 1
}


function test_tar() {
  local listing="./
./etc/
./etc/nsswitch.conf
./usr/
./usr/titi
./usr/bin/
./usr/bin/java -> /path/to/bin/java"
  for i in "" ".gz" ".bz2" ".xz"; do
    check_eq "$listing" "$(get_tar_listing test-tar-${i:1}.tar$i)"
    check_eq "-rwxr-xr-x" "$(get_tar_permission test-tar-${i:1}.tar$i ./usr/titi)"
    check_eq "-rw-r--r--" "$(get_tar_permission test-tar-${i:1}.tar$i ./etc/nsswitch.conf)"
    # Test merging tar files
    check_eq "$listing" "$(get_tar_listing test-tar-inclusion-${i:1}.tar)"
    check_eq "-rwxr-xr-x" "$(get_tar_permission test-tar-inclusion-${i:1}.tar ./usr/titi)"
    check_eq "-rw-r--r--" "$(get_tar_permission test-tar-inclusion-${i:1}.tar ./etc/nsswitch.conf)"
  done;

  check_eq "./
./nsswitch.conf" "$(get_tar_listing test-tar-strip_prefix-empty.tar)"
  check_eq "./
./nsswitch.conf" "$(get_tar_listing test-tar-strip_prefix-none.tar)"
  check_eq "./
./nsswitch.conf" "$(get_tar_listing test-tar-strip_prefix-etc.tar)"
  check_eq "./
./etc/
./etc/nsswitch.conf" "$(get_tar_listing test-tar-strip_prefix-dot.tar)"
}

function test_deb() {
  if ! (which dpkg-deb); then
    echo "Unable to run test for debian, no dpkg-deb!" >&2
    return 0
  fi
  local listing="./
./etc/
./etc/nsswitch.conf
./usr/
./usr/titi
./usr/bin/
./usr/bin/java -> /path/to/bin/java"
  check_eq "$listing" "$(get_deb_listing test-deb.deb)"
  check_eq "-rwxr-xr-x" "$(get_deb_permission test-deb.deb ./usr/titi)"
  check_eq "-rw-r--r--" "$(get_deb_permission test-deb.deb ./etc/nsswitch.conf)"
  get_deb_description test-deb.deb >$TEST_log
  expect_log "Description: toto"
  expect_log "Package: titi"
  expect_log "Depends: dep1, dep2"
}

run_suite "build_test"
