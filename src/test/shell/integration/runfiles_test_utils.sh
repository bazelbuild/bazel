#!/bin/bash
#
# Copyright 2022 The Bazel Authors. All rights reserved.
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

# Additional (path realFileType linkedToFileType) entries that the py_runtime
# adds to a target's runfiles. Google uses a custom runtime for integration
# tests, which add a different set of files.
function get_python_runtime_runfiles() {
  :;
}

# Additional repo mapping manifest file.
function get_repo_mapping_manifest_file() {
  echo ""
  echo "../repo_mapping file"
}
