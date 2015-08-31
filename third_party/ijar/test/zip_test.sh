#!/bin/bash -eu
#
# Copyright 2015 Google Inc. All rights reserved.
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

# Integration tests for ijar zipper/unzipper


DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

## Inputs
ZIPPER=${PWD}/$1
shift
UNZIP=$1
shift
ZIP=$1
shift

## Test framework
source ${DIR}/testenv.sh || { echo "testenv.sh not found!" >&2; exit 1; }

# Assertion
function assert_unzip_same_as_zipper() {
  local folder1=$(mktemp -d ${TEST_TMPDIR}/output.XXXXXXXX)
  local folder2=$(mktemp -d ${TEST_TMPDIR}/output.XXXXXXXX)
  (cd $folder1 && $UNZIP -q $1 || true)  # ignore CRC32 errors
  (cd $folder2 && $ZIPPER x $1)
  diff -r $folder1 $folder2 &> $TEST_log \
      || fail "Unzip and Zipper resulted in different output"
}

function assert_zipper_same_after_unzip() {
  local zipfile=${TEST_TMPDIR}/output.zip
  (cd $1 && $ZIPPER c ${zipfile} $(find . | sed 's|^./||' | grep -v '^.$'))
  local folder=$(mktemp -d ${TEST_TMPDIR}/output.XXXXXXXX)
  (cd $folder && $UNZIP -q ${zipfile} || true)  # ignore CRC32 errors
  diff -r $1 $folder &> $TEST_log \
      || fail "Unzip after zipper output differ"
  # Retry with compression
  (cd $1 && $ZIPPER cC ${zipfile} $(find . | sed 's|^./||' | grep -v '^.$'))
  local folder=$(mktemp -d ${TEST_TMPDIR}/output.XXXXXXXX)
  (cd $folder && $UNZIP -q ${zipfile} || true)  # ignore CRC32 errors
  diff -r $1 $folder &> $TEST_log \
      || fail "Unzip after zipper output differ"
}

#### Tests

function test_zipper() {
  mkdir -p ${TEST_TMPDIR}/test/path/to/some
  mkdir -p ${TEST_TMPDIR}/test/some/other/path
  touch ${TEST_TMPDIR}/test/path/to/some/empty_file
  echo "toto" > ${TEST_TMPDIR}/test/path/to/some/file
  echo "titi" > ${TEST_TMPDIR}/test/path/to/some/other_file
  chmod +x ${TEST_TMPDIR}/test/path/to/some/other_file
  echo "tata" > ${TEST_TMPDIR}/test/file
  assert_zipper_same_after_unzip ${TEST_TMPDIR}/test
  assert_unzip_same_as_zipper ${TEST_TMPDIR}/output.zip

  # Test flatten option
  (cd ${TEST_TMPDIR}/test && $ZIPPER cf ${TEST_TMPDIR}/output.zip \
      $(find . | sed 's|^./||' | grep -v '^.$'))
  $ZIPPER v ${TEST_TMPDIR}/output.zip >$TEST_log
  expect_log "file"
  expect_log "other_file"
  expect_not_log "path"
  expect_not_log "/"

  # Test adding leading garbage at the begining of the file (for
  # self-extractable binary).
  echo "abcdefghi" >${TEST_TMPDIR}/test.zip
  cat ${TEST_TMPDIR}/output.zip >>${TEST_TMPDIR}/test.zip
  $ZIPPER v ${TEST_TMPDIR}/test.zip >$TEST_log
  expect_log "file"
  expect_log "other_file"
  expect_not_log "path"
}

run_suite "zipper tests"
