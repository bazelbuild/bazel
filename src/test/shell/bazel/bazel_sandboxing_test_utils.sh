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

# namespaces which are used by the sandbox were introduced in 3.8, so
# test won't run on earlier kernels
function check_kernel_version {
  if [ "${PLATFORM-}" = "darwin" ]; then
    echo "Test will skip: sandbox is not yet supported on Darwin."
    exit 0
  fi
  MAJOR=$(uname -r | sed 's/^\([0-9]*\)\.\([0-9]*\)\..*/\1/')
  MINOR=$(uname -r | sed 's/^\([0-9]*\)\.\([0-9]*\)\..*/\2/')
  if [ $MAJOR -lt 3 ]; then
    echo "Test will skip: sandbox requires kernel >= 3.8; got $(uname -r)"
    exit 0
  fi
  if [ $MAJOR -eq 3 ] && [ $MINOR -lt 8 ]; then
    echo "Test will skip: sandbox requires kernel >= 3.8; got $(uname -r)"
    exit 0
  fi
}

# Some CI systems might deactivate sandboxing
function check_sandbox_allowed {
  mkdir -p test
  # Create a program that check if unshare(2) is allowed.
  cat <<'EOF' > test/test.c
#define _GNU_SOURCE
#include <sched.h>
int main() {
  return unshare(CLONE_NEWNS | CLONE_NEWUTS | CLONE_NEWIPC | CLONE_NEWUSER);
}
EOF
  cat <<'EOF' >test/BUILD
cc_test(name = "sandbox_enabled", srcs = ["test.c"], copts = ["-std=c99"])
EOF
  bazel test //test:sandbox_enabled || {
    echo "Sandboxing disabled, skipping..."
    return false
  }
}
