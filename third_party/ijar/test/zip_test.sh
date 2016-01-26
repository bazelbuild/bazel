#!/bin/bash -eu
#
# Copyright 2015 The Bazel Authors. All rights reserved.
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
  local dir="$1"
  shift
  local zipfile=${TEST_TMPDIR}/output.zip
  (cd "${dir}" && $ZIPPER c ${zipfile} "$@")
  local folder=$(mktemp -d ${TEST_TMPDIR}/output.XXXXXXXX)
  (cd $folder && $UNZIP -q ${zipfile} || true)  # ignore CRC32 errors
  diff -r "${dir}" $folder &> $TEST_log \
      || fail "Unzip after zipper output differ"
  # Retry with compression
  (cd "${dir}" && $ZIPPER cC ${zipfile} "$@")
  local folder=$(mktemp -d ${TEST_TMPDIR}/output.XXXXXXXX)
  (cd $folder && $UNZIP -q ${zipfile} || true)  # ignore CRC32 errors
  diff -r "${dir}" $folder &> $TEST_log \
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
  filelist="$(cd ${TEST_TMPDIR}/test && find . | sed 's|^./||' | grep -v '^.$')"

  assert_zipper_same_after_unzip ${TEST_TMPDIR}/test ${filelist}
  assert_unzip_same_as_zipper ${TEST_TMPDIR}/output.zip

  # Test @filelist format
  echo "${filelist}" >${TEST_TMPDIR}/test.content
  assert_zipper_same_after_unzip ${TEST_TMPDIR}/test @${TEST_TMPDIR}/test.content
  assert_unzip_same_as_zipper ${TEST_TMPDIR}/output.zip

  # Test flatten option
  (cd ${TEST_TMPDIR}/test && $ZIPPER cf ${TEST_TMPDIR}/output.zip ${filelist})
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

function test_zipper_compression() {
  echo -n > ${TEST_TMPDIR}/a
  for i in $(seq 1 1000); do
    echo -n "a" >> ${TEST_TMPDIR}/a
  done
  $ZIPPER cCf ${TEST_TMPDIR}/output.zip ${TEST_TMPDIR}/a
  local out_size=$(cat ${TEST_TMPDIR}/output.zip | wc -c | xargs)
  local in_size=$(cat ${TEST_TMPDIR}/a | wc -c | xargs)
  check_gt "${in_size}" "${out_size}" "Output size is greater than input size"

  rm -fr ${TEST_TMPDIR}/out
  mkdir -p ${TEST_TMPDIR}/out
  (cd ${TEST_TMPDIR}/out && $ZIPPER x ${TEST_TMPDIR}/output.zip)
  diff ${TEST_TMPDIR}/a ${TEST_TMPDIR}/out/a &> $TEST_log \
      || fail "Unzip using zipper after zipper output differ"

  rm -fr ${TEST_TMPDIR}/out
  mkdir -p ${TEST_TMPDIR}/out
  (cd ${TEST_TMPDIR}/out && $UNZIP -q ${TEST_TMPDIR}/output.zip)
  diff ${TEST_TMPDIR}/a ${TEST_TMPDIR}/out/a &> $TEST_log \
      || fail "Unzip after zipper output differ"
}

run_suite "zipper tests"
