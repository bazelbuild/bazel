#!/bin/bash
#
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

# Setting up the environment for Bazel release scripts test.

[ -z "$TEST_SRCDIR" ] && { echo "TEST_SRCDIR not set!" >&2; exit 1; }

# Load the unit-testing framework
source "${TEST_SRCDIR}/io_bazel/src/test/shell/unittest.bash" || \
  { echo "Failed to source unittest.bash" >&2; exit 1; }

# Commit at which we cut the master to do the test so we always take the git
# repository in a consistent state.
: ${MASTER_COMMIT:=7d41d7417fc34f7fa8aac7130a0588b8557e4b57}

# Set-up a copy of the git repository in ${MASTER_ROOT}, pointing master
# to ${MASTER_COMMIT}.
function setup_git_repository() {
  local origin_git_root=${TEST_SRCDIR}/io_bazel
  MASTER_ROOT=${TEST_TMPDIR}/git/root
  local orig_dir=${PWD}
  # Create a new origin with the good starting point
  mkdir -p ${MASTER_ROOT}
  cd ${MASTER_ROOT}
  cp -RL ${origin_git_root}/.git .git
  rm -f .git/hooks/*  # Do not keep custom hooks
  git reset -q --hard HEAD
  git checkout -q -B master ${MASTER_COMMIT}
  cd ${orig_dir}
}
