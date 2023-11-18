# Copyright 2023 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Filtered rule kinds for aspect inspection.

The format of this dictionary is:
  rule_name: [attr, attr, ...]

Filters for rules that are not part of the Bazel distribution should be added
to this file.

Attributes are either the explicit list of attributes to filter, or '_*' which
would ignore all attributes prefixed with a _.
"""

# Rule kinds with attributes the aspect currently needs to ignore
user_aspect_filters = {
}
