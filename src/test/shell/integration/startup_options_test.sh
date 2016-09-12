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
#
# Test of Bazel's startup option handling.

# Load test environment
source $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/testenv.sh \
  || { echo "testenv.sh not found!" >&2; exit 1; }

put_bazel_on_path
create_and_cd_client
write_default_bazelrc

function test_different_startup_options() {
  pid=$(bazel info server_pid 2> $TEST_log)
  [[ -n $pid ]] || fail "Couldn't run bazel"
  newpid=$(blaze --batch info server_pid 2> $TEST_log)
  expect_log "WARNING: Running B\\(azel\\|laze\\) server needs to be killed, because the startup options are different."
  [[ "$newpid" != "$pid" ]] || fail "pid $pid was the same!"
  kill -0 $pid 2> /dev/null && fail "$pid not dead"
  kill -0 $newpid 2> /dev/null && fail "$newpid not dead"
}

run_suite "bazel startup options test"
