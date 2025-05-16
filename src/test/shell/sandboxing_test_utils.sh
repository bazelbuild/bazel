#!/usr/bin/env bash
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

function check_sandbox_allowed {
  case "$(uname -s)" in
    Linux)
      if ! $linux_sandbox -- /bin/true; then
        echo "Skipping test: Sandboxing disabled or not supported" 2>&1
        return 1
      fi
      ;;

    *)
      return 0
      ;;
  esac
}
