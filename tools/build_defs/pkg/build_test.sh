#!/bin/bash
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

# Unit tests for pkg_deb and pkg_tar

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

function get_deb_ctl_file() {
  local input=$1
  local file=$2
  local test_data="${TEST_DATA_DIR}/${input}"
  dpkg-deb -I "${test_data}" "${file}"
}

function get_deb_ctl_listing() {
  local input=$1
  local test_data="${TEST_DATA_DIR}/${input}"
  dpkg-deb --ctrl-tarfile "${test_data}" | tar tf - | sort
}

function get_deb_ctl_file() {
  local input=$1
  local ctl_file=$2
  local test_data="${TEST_DATA_DIR}/${input}"
  dpkg-deb --info "${test_data}" "${ctl_file}"
}

function get_deb_ctl_permission() {
  local input=$1
  local file=$2
  local test_data="${TEST_DATA_DIR}/${input}"
  dpkg-deb --ctrl-tarfile "${test_data}" | tar tvf - | egrep " $file\$" | cut -d " " -f 1
}

function dpkg_deb_supports_ctrl_tarfile() {
  local input=$1
  local test_data="${TEST_DATA_DIR}/${input}"
  dpkg-deb --ctrl-tarfile "${test_data}" > /dev/null 2> /dev/null
}

function get_changes() {
  local input=$1
  cat "${TEST_DATA_DIR}/${input}"
}

function assert_content() {
  local listing="./
./etc/
./etc/nsswitch.conf
./usr/
./usr/titi
./usr/bin/
./usr/bin/java -> /path/to/bin/java"
  check_eq "$listing" "$(get_tar_listing $1)"
  check_eq "-rwxr-xr-x" "$(get_tar_permission $1 ./usr/titi)"
  check_eq "-rw-r--r--" "$(get_tar_permission $1 ./etc/nsswitch.conf)"
  check_eq "24/42" "$(get_numeric_tar_owner $1 ./etc/)"
  check_eq "24/42" "$(get_numeric_tar_owner $1 ./etc/nsswitch.conf)"
  check_eq "42/24" "$(get_numeric_tar_owner $1 ./usr/)"
  check_eq "42/24" "$(get_numeric_tar_owner $1 ./usr/titi)"
  if [ -z "${2-}" ]; then
    check_eq "tata/titi" "$(get_tar_owner $1 ./etc/)"
    check_eq "tata/titi" "$(get_tar_owner $1 ./etc/nsswitch.conf)"
    check_eq "titi/tata" "$(get_tar_owner $1 ./usr/)"
    check_eq "titi/tata" "$(get_tar_owner $1 ./usr/titi)"
  fi
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
    assert_content "test-tar-${i:1}.tar$i"
    # Test merging tar files
    # We pass a second argument to not test for user and group
    # names because tar merging ask for numeric owners.
    assert_content "test-tar-inclusion-${i:1}.tar" "true"
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
  check_eq "./
./not-etc/
./not-etc/mapped-filename.conf" "$(get_tar_listing test-tar-files_dict.tar)"
  check_eq "drwxr-xr-x 0/0               0 2000-01-01 00:00 ./
-rwxrwxrwx 0/0               0 2000-01-01 00:00 ./a
-rwxrwxrwx 0/0               0 2000-01-01 00:00 ./b" \
      "$(get_tar_verbose_listing test-tar-empty_files.tar)"
  check_eq "drwxr-xr-x 0/0               0 2000-01-01 00:00 ./
drwxrwxrwx 0/0               0 2000-01-01 00:00 ./tmp/
drwxrwxrwx 0/0               0 2000-01-01 00:00 ./pmt/" \
      "$(get_tar_verbose_listing test-tar-empty_dirs.tar)"
  check_eq \
    "drwxr-xr-x 0/0               0 1999-12-31 23:59 ./
-r-xr-xr-x 0/0               2 1999-12-31 23:59 ./nsswitch.conf" \
    "$(get_tar_verbose_listing test-tar-mtime.tar)"
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
  expect_log "Description: toto ®, Й, ק ,م, ๗, あ, 叶, 葉, 말, ü and é"
  expect_log "Package: titi"
  expect_log "soméone@somewhere.com"
  expect_log "Depends: dep1, dep2"
  expect_log "Built-Using: some_test_data"

  get_changes titi_test_all.changes >$TEST_log
  expect_log "Urgency: low"
  expect_log "Distribution: trusty"

  get_deb_ctl_file test-deb.deb templates >$TEST_log
  expect_log "Template: titi/test"
  expect_log "Type: string"

  get_deb_ctl_file test-deb.deb config >$TEST_log
  expect_log "# test config file"

  if ! dpkg_deb_supports_ctrl_tarfile test-deb.deb ; then
    echo "Unable to test deb control files listing, too old dpkg-deb!" >&2
    return 0
  fi
  local ctrl_listing="conffiles
config
control
templates"
  check_eq "$ctrl_listing" "$(get_deb_ctl_listing test-deb.deb)"
  check_eq "-rw-r--r--" "$(get_deb_ctl_permission test-deb.deb conffiles)"
  check_eq "-rwxr-xr-x" "$(get_deb_ctl_permission test-deb.deb config)"
  check_eq "-rw-r--r--" "$(get_deb_ctl_permission test-deb.deb control)"
  check_eq "-rwxr-xr-x" "$(get_deb_ctl_permission test-deb.deb templates)"
  local conffiles="/etc/nsswitch.conf
/etc/other"
  check_eq "$conffiles" "$(get_deb_ctl_file test-deb.deb conffiles)"
}

run_suite "build_test"
