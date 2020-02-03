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

'python_version' value for py_binary and py_test rules in the Bazel source tree.
See also "version.bzl.tools", which defines the value for targets in the
@bazel_tools repository.

We use PY3 in the source tree because PY2 reaches end of life in December 2019.
Building Bazel requires PY3.

We use PY2 in @bazel_tools to retain PY2 compatibility for users who run Bazel
with Python 2.
"""

PY_BINARY_VERSION = "PY3"
