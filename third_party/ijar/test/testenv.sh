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
#
# Setting up the environment for Bazel integration tests.
#

[ -z "$TEST_SRCDIR" ] && { echo "TEST_SRCDIR not set!" >&2; exit 1; }

# Load the unit-testing framework
source "${TEST_SRCDIR}/io_bazel/src/test/shell/unittest.bash" || \
  { echo "Failed to source unittest.bash" >&2; exit 1; }

## OSX/BSD stat and MD5
PLATFORM="$(uname -s | tr 'A-Z' 'a-z')"
if [[ "$PLATFORM" = "linux" ]]; then
  function statfmt() {
    stat -c "%s" $1
  }
  MD5SUM=md5sum
else
  function statfmt() {
    stat -f "%z" $1
  }
  MD5SUM=/sbin/md5
fi
