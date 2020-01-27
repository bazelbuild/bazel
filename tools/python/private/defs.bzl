# Copyright 2019 The Bazel Authors. All rights reserved.
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

"""This is a Bazel-internal file; do not load() it!

The incompatible change `--incompatible_load_python_rules_from_bzl` (#9006)
makes it so the four native Python rules cannot be used unless a magic tag is
present. It is intended that only `@rules_python` (bazelbuild/rules_python)
uses this tag, and all other uses access the native rules via the wrapper
macros defined in `@rules_python`.

However, `@bazel_tools` is not allowed to depend on any other repos. Therefore,
we replicate the behavior of `@rules_python`'s wrapper macros in this file, for
use by Bazel only.

This gets a bit tricky with the third_party/ directory. Some of its
subdirectories are full-blown workspaces that cannot directly reference this
file's label. For these cases we create a mock `@rules_python` in Bazel's
WORKSPACE, and rely on those repos using the proper @rules_python-qualified
label. (#9029 tracks possibly replacing the mock with the real thing.)
"""

_MIGRATION_TAG = "__PYTHON_RULES_MIGRATION_DO_NOT_USE_WILL_BREAK__"

def _add_tags(attrs):
    if "tags" in attrs and attrs["tags"] != None:
        attrs["tags"] += [_MIGRATION_TAG]
    else:
        attrs["tags"] = [_MIGRATION_TAG]
    return attrs

def py_library(**attrs):
    native.py_library(**_add_tags(attrs))

def py_binary(**attrs):
    native.py_binary(**_add_tags(attrs))

def py_test(**attrs):
    native.py_test(**_add_tags(attrs))

def py_runtime(**attrs):
    native.py_runtime(**_add_tags(attrs))
