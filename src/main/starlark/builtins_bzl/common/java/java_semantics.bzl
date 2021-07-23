# Copyright 2021 The Bazel Authors. All rights reserved.
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

"""
Java Semantics
"""

EXPERIMENTAL_USE_FILEGROUPS_IN_JAVALIBRARY = True
COLLECT_SRCS_FROM_PROTO_LIBRARY = False
EXTRA_SRCS_TYPES = []

ALLOWED_RULES_IN_DEPS = [
    "cc_binary",  # NB: linkshared=1
    "cc_library",
    "genrule",
    "genproto",  # TODO(bazel-team): we should filter using providers instead (starlark rule).
    "java_import",
    "java_library",
    "java_proto_library",
    "java_lite_proto_library",
    "proto_library",
    "sh_binary",
    "sh_library",
]

ALLOWED_RULES_IN_DEPS_WITH_WARNING = []
