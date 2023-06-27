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
Proto Semantics
"""

def _preprocess(ctx):
    pass

semantics = struct(
    PROTO_COMPILER_LABEL = "@bazel_tools//tools/proto:protoc",
    EXTRA_ATTRIBUTES = {
        "import_prefix": attr.string(),
    },
    EXTRA_FRAGMENTS = [],
    preprocess = _preprocess,
    # This constant is used in ProtoCompileActionBuilder to generate an error message that's
    # displayed when a strict proto deps violation occurs.
    #
    # %s is replaced with the label of the proto_library rule that's currently being built.
    # %%s is replaced with the literal "%s", which is passed to the proto-compiler, which replaces
    # it with the .proto file that violates strict proto deps.
    STRICT_DEPS_FLAG_TEMPLATE = (
        "--direct_dependencies_violation_msg=" +
        "%%s is imported, but %s doesn't directly depend on a proto_library that 'srcs' it."
    ),
    EXEC_GROUPS = {},
    allowlist_different_package = None,
)
