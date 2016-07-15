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

function check_supported_platform {
  if [ "${PLATFORM-}" = "darwin" ]; then
    echo "Test will skip: sandbox is not yet supported on Darwin."
    return 1
  fi
}

function check_sandbox_allowed {
  $linux_sandbox -C || {
    echo "Sandboxing disabled or not supported on this system, skipping..."
    return 1
  }
}
