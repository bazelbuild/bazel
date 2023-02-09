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
"""Attributes specific to the Bazel implementation of the Python rules."""

IMPORTS_ATTRS = {
    "imports": attr.string_list(
        doc = """
List of import directories to be added to the PYTHONPATH.

Subject to "Make variable" substitution. These import directories will be added
for this rule and all rules that depend on it (note: not the rules this rule
depends on. Each directory will be added to `PYTHONPATH` by `py_binary` rules
that depend on this rule. The strings are repo-runfiles-root relative,

Absolute paths (paths that start with `/`) and paths that references a path
above the execution root are not allowed and will result in an error.
""",
    ),
}
