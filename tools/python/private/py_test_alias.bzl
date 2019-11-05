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

"""Only code in @bazel_tools is allowed to load() this.

Defines an alias for py_test(). The Google-internal version of this rule is a
macro that generates a py_test for PY2 and PY3, to help migrating scripts.
Bazel's Python scripts don't need that macro, so we alias it to py_test.
"""

load(":private/defs.bzl", "py_test")

# TODO(bazel-team): delete this alias, replace with py_test everywhere as part
# of fixing https://github.com/bazelbuild/bazel/issues/10127
py_test_alias = py_test
