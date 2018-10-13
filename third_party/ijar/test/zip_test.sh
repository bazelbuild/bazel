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
  local zipfile=$1
  shift
  (cd $folder1 && $UNZIP -q $zipfile $@ || true)  # ignore CRC32 errors
  (cd $folder2 && $ZIPPER x $zipfile $@)
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
  expect_log "^f .* file$"
  expect_log "^f .* other_file$"
  expect_not_log "path"
  expect_not_log "/"

  # Test adding leading garbage at the begining of the file (for
  # self-extractable binary).
  echo "abcdefghi" >${TEST_TMPDIR}/test.zip
  cat ${TEST_TMPDIR}/output.zip >>${TEST_TMPDIR}/test.zip
  $ZIPPER v ${TEST_TMPDIR}/test.zip >$TEST_log
  expect_log "^f .* file$"
  expect_log "^f .* other_file$"
  expect_not_log "path"
}

function test_zipper_junk_paths() {
  mkdir -p ${TEST_TMPDIR}/test/path/to/some
  mkdir -p ${TEST_TMPDIR}/test/some/other/path
  touch ${TEST_TMPDIR}/test/path/to/some/empty_file
  echo "toto" > ${TEST_TMPDIR}/test/path/to/some/file
  echo "titi" > ${TEST_TMPDIR}/test/path/to/some/other_file
  chmod +x ${TEST_TMPDIR}/test/path/to/some/other_file
  echo "tata" > ${TEST_TMPDIR}/test/file
  filelist="$(cd ${TEST_TMPDIR}/test && find . | sed 's|^./||' | grep -v '^.$')"

  # Test extract + flatten option
  (cd ${TEST_TMPDIR}/test && $ZIPPER c ${TEST_TMPDIR}/output.zip ${filelist})
  $ZIPPER vf ${TEST_TMPDIR}/output.zip >$TEST_log
  echo $TEST_log
  expect_log "^f .* file$"
  expect_log "^f .* other_file$"
  expect_not_log "path"
  expect_not_log "/"
}

function test_zipper_unzip_selective_files() {
  mkdir -p ${TEST_TMPDIR}/test/path/to/some
  mkdir -p ${TEST_TMPDIR}/test/some/other/path
  touch ${TEST_TMPDIR}/test/path/to/some/empty_file
  echo "toto" > ${TEST_TMPDIR}/test/path/to/some/file
  echo "titi" > ${TEST_TMPDIR}/test/path/to/some/other_file
  chmod +x ${TEST_TMPDIR}/test/path/to/some/other_file
  echo "tata" > ${TEST_TMPDIR}/test/file
  filelist="$(cd ${TEST_TMPDIR}/test && find . | sed 's|^./||' | grep -v '^.$')"

  assert_zipper_same_after_unzip ${TEST_TMPDIR}/test ${filelist}
  assert_unzip_same_as_zipper ${TEST_TMPDIR}/output.zip \
      path/to/some/empty_file path/to/some/other_file
}

function test_zipper_unzip_to_optional_dir() {
  mkdir -p ${TEST_TMPDIR}/test/path/to/some
  mkdir -p ${TEST_TMPDIR}/test/some/other/path
  touch ${TEST_TMPDIR}/test/path/to/some/empty_file
  echo "toto" > ${TEST_TMPDIR}/test/path/to/some/file
  echo "titi" > ${TEST_TMPDIR}/test/path/to/some/other_file
  chmod +x ${TEST_TMPDIR}/test/path/to/some/other_file
  echo "tata" > ${TEST_TMPDIR}/test/file
  filelist="$(cd ${TEST_TMPDIR}/test && find . | sed 's|^./||' | grep -v '^.$')"

  assert_zipper_same_after_unzip ${TEST_TMPDIR}/test ${filelist}
  assert_unzip_same_as_zipper ${TEST_TMPDIR}/output.zip -d output_dir2 \
      path/to/some/empty_file path/to/some/other_file
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

function test_zipper_specify_path() {
  mkdir -p ${TEST_TMPDIR}/files
  echo "toto" > ${TEST_TMPDIR}/files/a.txt
  echo "titi" > ${TEST_TMPDIR}/files/b.txt
  rm -fr ${TEST_TMPDIR}/expect/foo/bar
  mkdir -p ${TEST_TMPDIR}/expect/foo/bar
  touch ${TEST_TMPDIR}/expect/empty.txt
  echo "toto" > ${TEST_TMPDIR}/expect/foo/a.txt
  echo "titi" > ${TEST_TMPDIR}/expect/foo/bar/b.txt
  rm -fr ${TEST_TMPDIR}/out
  mkdir -p ${TEST_TMPDIR}/out

  ${ZIPPER} cC ${TEST_TMPDIR}/output.zip empty.txt= \
      foo/a.txt=${TEST_TMPDIR}/files/a.txt \
      foo/bar/b.txt=${TEST_TMPDIR}/files/b.txt
  (cd ${TEST_TMPDIR}/out && $UNZIP -q ${TEST_TMPDIR}/output.zip)
  diff -r ${TEST_TMPDIR}/expect ${TEST_TMPDIR}/out &> $TEST_log \
      || fail "Unzip after zipper output is not expected"
}

function test_zipper_permissions() {
  local -r LOCAL_TEST_DIR="${TEST_TMPDIR}/${FUNCNAME[0]}"
  mkdir -p ${LOCAL_TEST_DIR}/files
  printf "#!/bin/sh\nexit 0\n" > ${LOCAL_TEST_DIR}/files/executable
  printf "#!/bin/sh\nexit 0\n" > ${LOCAL_TEST_DIR}/files/non_executable
  chmod +x ${LOCAL_TEST_DIR}/files/executable
  chmod -x ${LOCAL_TEST_DIR}/files/non_executable

  ${ZIPPER} cC ${LOCAL_TEST_DIR}/output.zip \
      executable=${LOCAL_TEST_DIR}/files/executable \
      non_executable=${LOCAL_TEST_DIR}/files/non_executable

  mkdir -p ${LOCAL_TEST_DIR}/out
  cd ${LOCAL_TEST_DIR}/out && $UNZIP -q ${LOCAL_TEST_DIR}/output.zip

  if ! test -x ${LOCAL_TEST_DIR}/out/executable; then
    fail "out/executable should have been executable"
  fi
  if test -x ${LOCAL_TEST_DIR}/out/non_executable; then
    fail "out/non_executable should not have been executable"
  fi
}

run_suite "zipper tests"
