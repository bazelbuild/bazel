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
"""Contains constants that vary between Bazel and Google-internal"""

IMPORTS_ATTR_SUPPORTED = True

TOOLS_REPO = "bazel_tools"
PLATFORMS_LOCATION = "@platforms"

SRCS_ATTR_ALLOW_FILES = [".py", ".py3"]

DEPS_ATTR_ALLOW_RULES = None

PY_RUNTIME_ATTR_NAME = "_py_interpreter"
PY_RUNTIME_FRAGMENT_NAME = "py"
PY_RUNTIME_FRAGMENT_ATTR_NAME = "python_path"

BUILD_DATA_SYMLINK_PATH = "pyglib/build_data.txt"
